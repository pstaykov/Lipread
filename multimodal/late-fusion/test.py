"""
Evaluate the 30-word late-fusion model on 100 random test-split samples.
Injects additive white Gaussian noise at SNR = {clean, +10, +5, 0, -5, -10} dB.

Reports top-1 accuracy for three ablations:
  multimodal  — visual + audio (full model)
  visual-only — zero out audio features before MetaLearner
  audio-only  — zero out visual features before MetaLearner

Results are saved to test_results.json for the analysis notebook.

Usage:
  python test.py                       # expects checkpoints_15/best_meta.pth
  python test.py --ckpt path/to/meta.pth
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import _load_video_audio
from models import MetaLearner, VideoAugment, load_visual_encoder, load_audio_encoder

try:
    import whisper as whisper_lib
except ImportError:
    raise ImportError("openai-whisper not found: pip install openai-whisper")

NUM_WORDS      = 30
SUBSET_SEED    = 42
TEST_SAMPLES   = 100
TEST_SEED      = 123
SNR_CONDITIONS = [None, 10, 5, 0, -5, -10]  # None = clean


def add_awgn(wav: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to wav to achieve the target SNR (dB)."""
    signal_power = np.mean(wav.astype(np.float64) ** 2)
    if signal_power < 1e-10:
        return wav
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(wav)) * np.sqrt(noise_power)
    return (wav + noise).astype(np.float32)


class TestDataset(Dataset):
    def __init__(self, samples, snr_db=None, num_frames=25):
        self.samples = samples
        self.snr_db = snr_db
        self.num_frames = num_frames
        self.transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, audio_wav = _load_video_audio(video_path)

        if self.snr_db is not None:
            audio_wav = add_awgn(audio_wav, self.snr_db)

        video = video.float() / 255.0
        T = video.size(0)
        if T > self.num_frames:
            indices = np.linspace(0, T - 1, num=self.num_frames).astype(int)
            video = video[indices]
        elif T < self.num_frames:
            pad = self.num_frames - T
            video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)

        video = self.transform(video)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        mel = whisper_lib.log_mel_spectrogram(whisper_lib.pad_or_trim(audio_wav))

        return video, mel, label


@torch.no_grad()
def evaluate(loader, visual_enc, audio_enc, meta, device):
    """Returns (multimodal_acc, visual_only_acc, audio_only_acc)."""
    correct_mm = correct_v = correct_a = total = 0

    for video, mel, labels in tqdm(loader, desc='  eval', leave=False):
        video  = video.to(device, non_blocking=True)
        mel    = mel.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        v_feat = visual_enc(video)
        a_feat = audio_enc(mel).mean(dim=1)

        logits_mm = meta(v_feat, a_feat)
        logits_v  = meta(v_feat, torch.zeros_like(a_feat))
        logits_a  = meta(torch.zeros_like(v_feat), a_feat)

        correct_mm += (logits_mm.argmax(dim=1) == labels).sum().item()
        correct_v  += (logits_v.argmax(dim=1)  == labels).sum().item()
        correct_a  += (logits_a.argmax(dim=1)  == labels).sum().item()
        total += labels.size(0)

    return (correct_mm / total, correct_v / total, correct_a / total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None,
                        help='Path to meta-learner checkpoint (.pth). '
                             'Defaults to checkpoints_15/best_meta.pth then final_meta.pth.')
    parser.add_argument('--samples', type=int, default=TEST_SAMPLES,
                        help='Number of random test samples to evaluate (default: 100)')
    args = parser.parse_args()

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'lipreading', 'GLips', 'lipread_files')
    this_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")

    if args.ckpt:
        meta_ckpt = args.ckpt
    else:
        meta_ckpt = os.path.join(this_dir, 'checkpoints_15', 'best_meta.pth')
        if not os.path.exists(meta_ckpt):
            meta_ckpt = os.path.join(this_dir, 'checkpoints_15', 'final_meta.pth')
    if not os.path.exists(meta_ckpt):
        raise FileNotFoundError(
            f"No meta-learner checkpoint found. Train first with train_30.py.\n"
            f"Expected: {meta_ckpt}"
        )
    print(f"Meta checkpoint: {meta_ckpt}")

    all_classes = sorted([d for d in os.listdir(root_dir)
                          if os.path.isdir(os.path.join(root_dir, d))])
    rng = random.Random(SUBSET_SEED)
    selected_classes = sorted(rng.sample(all_classes, NUM_WORDS))
    class_to_idx = {c: i for i, c in enumerate(selected_classes)}
    print(f"30-word subset: {selected_classes}")

    all_test = []
    for cls_name in selected_classes:
        test_folder = os.path.join(root_dir, cls_name, 'test')
        if not os.path.exists(test_folder):
            test_folder = os.path.join(root_dir, cls_name, 'val')
        if os.path.exists(test_folder):
            for f in os.listdir(test_folder):
                if f.endswith('.mp4'):
                    all_test.append((os.path.join(test_folder, f), class_to_idx[cls_name]))

    rng2 = random.Random(TEST_SEED)
    test_subset = rng2.sample(all_test, min(args.samples, len(all_test)))
    print(f"Test samples: {len(test_subset)} (from {len(all_test)} available)")

    visual_ckpt = os.path.join(this_dir, '..', '..', 'lipreading', 'checkpoints', 'final_model.pth')
    print("Loading visual encoder...")
    visual_enc = load_visual_encoder(visual_ckpt, device)

    print("Loading Whisper-small audio encoder...")
    audio_enc = load_audio_encoder(device)

    print("Loading MetaLearner...")
    meta = MetaLearner(num_classes=NUM_WORDS).to(device)
    meta.load_state_dict(torch.load(meta_ckpt, map_location=device))
    meta.eval()

    num_workers = min(2, os.cpu_count() or 0)
    results = {}

    print(f"\n{'Condition':<12}  {'Multimodal':>12}  {'Visual-only':>12}  {'Audio-only':>12}")
    print('-' * 52)

    for snr in SNR_CONDITIONS:
        label = 'clean' if snr is None else f'{snr:+d}dB'
        dataset = TestDataset(test_subset, snr_db=snr)
        loader  = DataLoader(dataset, batch_size=8, shuffle=False,
                             num_workers=num_workers, pin_memory=device.type == 'cuda')

        acc_mm, acc_v, acc_a = evaluate(loader, visual_enc, audio_enc, meta, device)
        results[label] = {
            'multimodal':  round(acc_mm * 100, 2),
            'visual_only': round(acc_v  * 100, 2),
            'audio_only':  round(acc_a  * 100, 2),
        }
        print(f"{label:<12}  {acc_mm*100:>11.1f}%  {acc_v*100:>11.1f}%  {acc_a*100:>11.1f}%")

    out_path = os.path.join(this_dir, 'test_results.json')
    with open(out_path, 'w') as f:
        json.dump({'snr_conditions': SNR_CONDITIONS, 'results': results,
                   'num_samples': len(test_subset), 'num_classes': NUM_WORDS}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
