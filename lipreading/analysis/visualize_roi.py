"""Visualize the real mouth-ROI pipeline for a few GLips-15 classes, side by side:
raw frame + FaceMesh landmarks + crop box | actual 96x96 mouth-ROI crop | actual 88x88 model input,
plus a temporal strip of the model input to confirm the lips are present and move.

Run from the lipreading/ directory:
    python analysis/visualize_roi.py
Outputs to analysis/roi_check/.
"""
import os
import sys

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# dataset.py / dataset15.py / preprocess_mouth_roi.py live one level up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import VideoAugment, _load_video_audio  # noqa: E402
from dataset15 import CLASSES, FULL_ROOT, ROI_ROOT  # noqa: E402
from preprocess_mouth_roi import LIP_IDX, _face_mesh, lip_box, smooth  # noqa: E402
import mediapipe as mp  # noqa: E402

OUT_DIR = './analysis/roi_check'
RESIZE = 96
CROP = 88
BOX_SCALE = 1.6   # matches preprocess_mouth_roi.py's default --box-scale
ALPHA = 0.4       # matches preprocess_mouth_roi.py's default --alpha
N_SAMPLES = 6     # one clip from each of the first N GLips-15 classes
STRIP_FRAMES = 10  # frames in the temporal strip

LIP_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_LIPS

# ImageNet stats used by VideoAugment — needed to undo normalization for display
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(chw):
    """Undo ImageNet normalization on a (C,H,W) tensor -> uint8-friendly float [0,1]."""
    return (chw * _STD + _MEAN).clamp(0, 1)


def pick_sample(root, cls):
    """Return (path, split) to one .mp4 for `cls`, preferring the train split."""
    for split in ('train', 'val', 'validation'):
        folder = os.path.join(root, cls, split)
        if os.path.isdir(folder):
            mp4s = sorted(f for f in os.listdir(folder) if f.endswith('.mp4'))
            if mp4s:
                return os.path.join(folder, mp4s[0]), split
    return None, None


def real_box_and_landmarks(path, box_scale=BOX_SCALE, alpha=ALPHA):
    """Replay preprocess_mouth_roi.py's per-frame EMA box computation up to the middle frame; returns (mid_frame_bgr, box, landmarks_or_None)."""
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    mid = n_frames // 2
    fm = _face_mesh()
    last_box, mid_frame, mid_landmarks = None, None, None
    for i in range(mid + 1):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            landmarks = res.multi_face_landmarks[0].landmark
            last_box = smooth(last_box, lip_box(landmarks, w, h, box_scale), alpha)
            if i == mid:
                mid_landmarks = landmarks
        elif last_box is None:
            last_box = (w / 2, h / 2, min(w, h) * 0.4)
        if i == mid:
            mid_frame = frame
    cap.release()
    return mid_frame, last_box, mid_landmarks


def draw_landmarks(ax, landmarks, w, h):
    """Overlay all FaceMesh points (faint) with the lip subset + contour highlighted."""
    if landmarks is None:
        return
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    ax.scatter(xs, ys, s=1, c='cyan', alpha=0.25, linewidths=0)

    for a, b in LIP_CONNECTIONS:
        ax.plot([landmarks[a].x * w, landmarks[b].x * w],
                 [landmarks[a].y * h, landmarks[b].y * h],
                 c='red', linewidth=0.8, alpha=0.8)
    lip_xs = [landmarks[i].x * w for i in LIP_IDX]
    lip_ys = [landmarks[i].y * h for i in LIP_IDX]
    ax.scatter(lip_xs, lip_ys, s=6, c='red', alpha=0.9, linewidths=0)


def main():
    if not os.path.isdir(FULL_ROOT):
        raise SystemExit(f"Data path not found: {FULL_ROOT} (run from the lipreading/ dir)")
    if not os.path.isdir(ROI_ROOT):
        raise SystemExit(f"Mouth-ROI path not found: {ROI_ROOT} — run preprocess_mouth_roi.py first")
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = CLASSES[:N_SAMPLES]
    val_aug = VideoAugment(crop_size=CROP, resize_size=RESIZE, is_train=False)

    fig, axes = plt.subplots(len(classes), 3, figsize=(9, 3 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]

    strips = []  # (class, split, processed model-input clip) collected for figure 2
    for r, cls in enumerate(classes):
        raw_path, split = pick_sample(FULL_ROOT, cls)
        if raw_path is None:
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"no clip\n{cls}", ha='center', va='center')
                axes[r, c].axis('off')
            continue

        rel = os.path.relpath(raw_path, FULL_ROOT)
        mouth_path = os.path.join(ROI_ROOT, rel)
        if not os.path.isfile(mouth_path):
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"no mouth-ROI clip\n{cls}", ha='center', va='center')
                axes[r, c].axis('off')
            continue

        mid_frame_bgr, box, landmarks = real_box_and_landmarks(raw_path)
        h, w = mid_frame_bgr.shape[:2]
        axes[r, 0].imshow(cv2.cvtColor(mid_frame_bgr, cv2.COLOR_BGR2RGB))
        draw_landmarks(axes[r, 0], landmarks, w, h)
        cx, cy, side = box
        axes[r, 0].add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                       fill=False, edgecolor='lime', lw=2))
        axes[r, 0].set_title(f"{cls} [{split}] raw {w}x{h} + landmarks", fontsize=9)

        mouth_raw, _ = _load_video_audio(mouth_path)     # (T,C,96,96) uint8
        mouth_mid = mouth_raw[mouth_raw.shape[0] // 2].float() / 255.0
        axes[r, 1].imshow(mouth_mid.permute(1, 2, 0).numpy())
        axes[r, 1].set_title(f"actual mouth-ROI crop {mouth_raw.shape[-1]}x{mouth_raw.shape[-1]}",
                             fontsize=9)

        mouth_vid01 = mouth_raw.float() / 255.0
        processed = val_aug(mouth_vid01)                 # (T,C,CROP,CROP), normalized
        proc_mid = denorm(processed[processed.shape[0] // 2])
        axes[r, 2].imshow(proc_mid.permute(1, 2, 0).numpy())
        axes[r, 2].set_title(f"actual model input {CROP}x{CROP}", fontsize=9)

        for c in range(3):
            axes[r, c].axis('off')

        strips.append((cls, split, processed))

    fig.suptitle("Lip-ROI check: red = FaceMesh lip landmarks, lime = crop box they produce",
                fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    overview_path = os.path.join(OUT_DIR, 'roi_overview.png')
    fig.savefig(overview_path, dpi=110)
    plt.close(fig)
    print(f"wrote {overview_path}")

    if strips:
        fig2, axes2 = plt.subplots(len(strips), STRIP_FRAMES,
                                   figsize=(1.4 * STRIP_FRAMES, 1.5 * len(strips)))
        axes2 = np.atleast_2d(axes2)
        for r, (cls, split, processed) in enumerate(strips):
            T = processed.shape[0]
            idxs = np.linspace(0, T - 1, STRIP_FRAMES).astype(int)
            for c, t in enumerate(idxs):
                axes2[r, c].imshow(denorm(processed[t]).permute(1, 2, 0).numpy())
                axes2[r, c].axis('off')
                if c == 0:
                    axes2[r, c].set_ylabel(cls, fontsize=9)
            axes2[r, 0].set_title(f"{cls} (t over clip ->)", fontsize=8, loc='left')
        fig2.suptitle("Temporal strip of actual model input (lips should be centered and move)",
                     fontsize=12)
        fig2.tight_layout(rect=[0, 0, 1, 0.97])
        strip_path = os.path.join(OUT_DIR, 'roi_temporal.png')
        fig2.savefig(strip_path, dpi=110)
        plt.close(fig2)
        print(f"wrote {strip_path}")

    print("Done. Open the PNGs in", OUT_DIR)


if __name__ == '__main__':
    main()
