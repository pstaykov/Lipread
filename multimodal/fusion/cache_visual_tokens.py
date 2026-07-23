"""Cache frozen visual token sequences, so every fusion head trains on identical
visual inputs without repeating the slow video-decode pass.

Runs the standalone visual model's trunk (cnn -> proj -> ms_tcn -> +pos ->
transformer) over every train/val clip and stores the pre-pool (Tv=25, Dv=256)
token sequence in a memmap keyed by the same class/split/name key the audio cache
uses, so a clip's visual tokens and cached waveform line up by key.

Clips are enumerated via the pipeline's own MultimodalGLipsDataset (split
'train'/'validation'), so the cached set is byte-identical to the one the SNR sweep
and audio probe use (val = 24900), keeping every downstream number comparable.

One-time cost is the video decode (~1h for train). After this, fusion training reads
tokens from the memmap and encodes audio live from the waveform cache (~3 min/pass).
"""
import os
import sys
import json
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
LIPREAD = os.path.abspath(os.path.join(HERE, '..', '..', 'lipreading'))
sys.path.insert(0, os.path.join(LIPREAD, 'Transformer_based'))
if 'mmtrain' not in sys.modules:
    _spec = importlib.util.spec_from_file_location('mmtrain', os.path.join(HERE, 'train.py'))
    m = importlib.util.module_from_spec(_spec)
    sys.modules['mmtrain'] = m
    _spec.loader.exec_module(m)
else:
    m = sys.modules['mmtrain']
from model import GLipsNet, _strip_orig_mod, D_MODEL   # noqa: E402  (Transformer_based/model.py)

ROOT = os.environ.get('VIS_ROOT', os.path.join(LIPREAD, 'GLips_mouth', 'lipread_files'))
AUDIO_ROOT = os.path.join(LIPREAD, 'GLips', 'lipread_files')
AUD_CACHE = os.path.join(HERE, 'cache', 'audio_cache')
CKPT = os.path.join(LIPREAD, 'Transformer_based', 'checkpoints', 'best_model.pth')
OUT_DIR = os.environ.get('VIS_CACHE', os.path.join(HERE, 'cache', 'visual_token_cache'))
EXCLUDED = ('hier', 'soll')
TV, DV = 25, D_MODEL
BATCH = 32
LIMIT = int(os.environ.get('LIMIT', '0'))   # >0 caps clips/split for a smoke test


@torch.no_grad()
def trunk_tokens(model, video):
    """GLipsNet.forward up to (but not including) pooling -> (B, Tv, D_MODEL)."""
    x = model.cnn(video)
    B, T, C, H, W = x.size()
    x = model.avgpool(x.view(B * T, C, H, W)).flatten(1).view(B, T, -1)
    x = model.proj(x)
    x = model.ms_tcn(x)
    x = x + model.pos_embed[:, :T, :]
    x = model.transformer(x)
    return x


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda')
    amp = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    classes = sorted(d for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d)) and d not in EXCLUDED)
    assert len(classes) == 498, f'expected 498 classes, got {len(classes)}'

    model = GLipsNet(num_classes=len(classes), pool='attn').to(device)
    model.load_state_dict(_strip_orig_mod(torch.load(CKPT, map_location=device, weights_only=True)))
    model.eval()
    print(f'loaded visual trunk from {CKPT}; caching from {ROOT}')

    tf = m.VideoAugment(crop_size=88, resize_size=96, is_train=False)
    splits = os.environ.get('SPLITS', 'train,validation').split(',')
    for split in splits:
        ds = m.MultimodalGLipsDataset(ROOT, split=split, num_frames=TV, transform=tf,
                                      audio_root=AUDIO_ROOT, cache_dir=AUD_CACHE,
                                      group_split=False, classes=classes)
        if LIMIT and len(ds.samples) > LIMIT:
            ds.samples = ds.samples[:LIMIT]
        n = len(ds)
        tag = 'val' if split == 'validation' else split
        print(f'{tag}: {n} clips')
        if n == 0:
            continue

        feats = np.memmap(os.path.join(OUT_DIR, f'{tag}_tokens.dat'), dtype=np.float16,
                          mode='w+', shape=(n, TV, DV))
        labels = np.empty(n, dtype=np.int64)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True, collate_fn=m.collate_fn)
        i = 0
        for video, _wav, target in tqdm(loader, desc=f'cache {tag}', unit='batch'):
            video = video.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=amp):
                tok = trunk_tokens(model, video)
            b = tok.size(0)
            feats[i:i + b] = tok.float().half().cpu().numpy()
            labels[i:i + b] = target.numpy()
            i += b
        feats.flush()

        rows = {m._audio_key(vp, ds.video_root): j for j, (vp, _) in enumerate(ds.samples)}
        with open(os.path.join(OUT_DIR, f'{tag}_index.json'), 'w') as f:
            json.dump({'n': n, 'Tv': TV, 'Dv': DV, 'rows': rows, 'classes': classes}, f)
        np.save(os.path.join(OUT_DIR, f'{tag}_labels.npy'), labels)
        print(f'  wrote {tag}_tokens.dat  ({n}x{TV}x{DV} fp16)')

    print('done.')


if __name__ == '__main__':
    main()
