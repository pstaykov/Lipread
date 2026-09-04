"""Held-out test-set evaluation for the fine-tuned 500-class multimodal fusion model.

    python evaluate_500_test.py                 # test split
    python evaluate_500_test.py --split val
"""
import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lipreading', 'analysis'))
from train import (  # noqa: E402
    GLipsNet, MultimodalGLipsDataset, VideoAugment, WhisperExtractor,
    collate_fn, _strip_orig_mod, WHISPER_MODEL_NAME,
)
from bootstrap_ci import bootstrap_ci  # noqa: E402

ROOT_DIR = '../lipreading/GLips_mouth/lipread_files'
AUDIO_ROOT = '../lipreading/GLips/lipread_files'
AUDIO_CACHE_DIR = './cache/audio_cache'
NUM_FRAMES = 25


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test', 'val', 'validation', 'train'])
    ap.add_argument('--checkpoint', default='./models/checkpoints_500/best_model.pth')
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extractor = WhisperExtractor(WHISPER_MODEL_NAME, device=device)
    val_tf = VideoAugment(crop_size=88, resize_size=96, is_train=False)

    CLASSES = sorted(d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d)))
    n = len(CLASSES)

    audio_cache = AUDIO_CACHE_DIR if os.path.exists(os.path.join(AUDIO_CACHE_DIR, 'index.json')) else None
    dataset = MultimodalGLipsDataset(ROOT_DIR, split=args.split, num_frames=NUM_FRAMES,
                                     transform=val_tf, audio_root=AUDIO_ROOT,
                                     cache_dir=audio_cache, group_split=False, classes=CLASSES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)
    print(f"Eval split='{args.split}'  classes={n}  samples={len(dataset)}  (stock split)")

    model = GLipsNet(num_classes=n, use_audio=True)
    model.load_state_dict(_strip_orig_mod(torch.load(args.checkpoint, map_location=device)))
    model.to(device).eval()

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    k5 = min(5, n)
    correct1, correct5 = [], []
    for video, wav, target in tqdm(loader, desc='Test', dynamic_ncols=True):
        video = video.to(device, non_blocking=True)
        wav = wav.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        audio = extractor.encode(wav)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            logits = model(video, audio)
        correct1.append((logits.argmax(dim=1) == target).cpu().numpy())
        correct5.append((logits.topk(k5, dim=1).indices == target.unsqueeze(1)).any(dim=1).cpu().numpy())
    correct1 = np.concatenate(correct1)
    correct5 = np.concatenate(correct5)
    total = correct1.shape[0]

    m1, lo1, hi1, hw1 = bootstrap_ci(correct1)
    m5, lo5, hi5, hw5 = bootstrap_ci(correct5)
    print(f"\n=== RESULTS ({args.split}, {total} samples, chance={1/n:.4f}) ===")
    print(f"checkpoint={args.checkpoint}")
    print(f"top1={m1:.4f}  top5={m5:.4f}")
    print(f"top1 95% bootstrap CI: [{lo1:.4f}, {hi1:.4f}]  (+/-{hw1:.4f}, n_boot=10000)")
    print(f"top5 95% bootstrap CI: [{lo5:.4f}, {hi5:.4f}]  (+/-{hw5:.4f}, n_boot=10000)")


if __name__ == '__main__':
    main()
