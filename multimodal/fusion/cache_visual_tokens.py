"""Cache frozen visual trunk token sequences (pre-pool) to a memmap keyed like the audio cache."""
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
CKPT = os.path.join(LIPREAD, 'Transformer_based', 'checkpoints_500', 'best_model.pth')
OUT_DIR = os.environ.get('VIS_CACHE', os.path.join(HERE, 'cache', 'visual_token_cache'))
EXCLUDED = ()  # hier/soll mouth-ROI bug fixed upstream; full 500-class vocabulary now
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
    assert len(classes) == 500, f'expected 500 classes, got {len(classes)}'

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
        ds_all_samples = list(ds.samples)
        n = len(ds)
        tag = 'val' if split == 'validation' else split
        print(f'{tag}: {n} clips')
        if n == 0:
            continue

        # resumable: `{tag}_done.txt` holds rows already written; memmap reopened 'r+' if it exists
        done_path = os.path.join(OUT_DIR, f'{tag}_done.txt')
        tokens_path = os.path.join(OUT_DIR, f'{tag}_tokens.dat')
        labels_path = os.path.join(OUT_DIR, f'{tag}_labels_wip.npy')
        start = 0
        if os.path.exists(done_path) and os.path.exists(tokens_path):
            with open(done_path) as f:
                start = int(f.read().strip() or 0)
            start = min(start, n)

        mode = 'r+' if start > 0 else 'w+'
        feats = np.memmap(tokens_path, dtype=np.float16, mode=mode, shape=(n, TV, DV))
        if start > 0 and os.path.exists(labels_path):
            labels = np.load(labels_path)
        else:
            labels = np.empty(n, dtype=np.int64)

        if start >= n:
            print(f'  {tag}: already complete ({n}/{n}), skipping to index write')
        else:
            print(f'  {tag}: resuming at {start}/{n}' if start else f'  {tag}: starting fresh')
            remaining = ds.samples[start:]
            ds.samples = remaining
            loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4,
                                pin_memory=True, persistent_workers=True, collate_fn=m.collate_fn)
            i = start
            for bi, (video, _wav, target) in enumerate(tqdm(loader, desc=f'cache {tag}', unit='batch')):
                video = video.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp):
                    tok = trunk_tokens(model, video)
                b = tok.size(0)
                feats[i:i + b] = tok.float().half().cpu().numpy()
                labels[i:i + b] = target.numpy()
                i += b
                if bi % 20 == 0:
                    feats.flush()
                    np.save(labels_path, labels)
                    with open(done_path, 'w') as f:
                        f.write(str(i))
            feats.flush()
            np.save(labels_path, labels)
            with open(done_path, 'w') as f:
                f.write(str(i))

        rows = {m._audio_key(vp, ds.video_root): j for j, (vp, _) in enumerate(ds_all_samples)}
        with open(os.path.join(OUT_DIR, f'{tag}_index.json'), 'w') as f:
            json.dump({'n': n, 'Tv': TV, 'Dv': DV, 'rows': rows, 'classes': classes}, f)
        np.save(os.path.join(OUT_DIR, f'{tag}_labels.npy'), labels)
        print(f'  wrote {tag}_tokens.dat  ({n}x{TV}x{DV} fp16)')

    print('done.')


if __name__ == '__main__':
    main()
