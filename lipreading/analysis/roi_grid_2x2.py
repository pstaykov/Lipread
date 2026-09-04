"""2x2 lip-ROI check grid: left column raw frame + landmarks + crop box, right column the actual mouth-ROI crop.

Run from the lipreading/ directory:
    python analysis/roi_grid_2x2.py
Outputs analysis/roi_check/roi_grid_2x2.png
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import _load_video_audio  # noqa: E402
from dataset15 import CLASSES, FULL_ROOT, ROI_ROOT  # noqa: E402
from visualize_roi import pick_sample, real_box_and_landmarks, draw_landmarks  # noqa: E402

OUT_DIR = './analysis/roi_check'
N_SAMPLES = 2


def main():
    if not os.path.isdir(FULL_ROOT):
        raise SystemExit(f"Data path not found: {FULL_ROOT} (run from the lipreading/ dir)")
    if not os.path.isdir(ROI_ROOT):
        raise SystemExit(f"Mouth-ROI path not found: {ROI_ROOT} — run preprocess_mouth_roi.py first")
    os.makedirs(OUT_DIR, exist_ok=True)

    classes = CLASSES[:N_SAMPLES]
    fig, axes = plt.subplots(len(classes), 2, figsize=(6, 3 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]

    for r, cls in enumerate(classes):
        raw_path, split = pick_sample(FULL_ROOT, cls)
        rel = os.path.relpath(raw_path, FULL_ROOT)
        mouth_path = os.path.join(ROI_ROOT, rel)

        mid_frame_bgr, box, landmarks = real_box_and_landmarks(raw_path)
        h, w = mid_frame_bgr.shape[:2]
        axes[r, 0].imshow(cv2.cvtColor(mid_frame_bgr, cv2.COLOR_BGR2RGB))
        draw_landmarks(axes[r, 0], landmarks, w, h)
        cx, cy, side = box
        axes[r, 0].add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                       fill=False, edgecolor='lime', lw=2))

        mouth_raw, _ = _load_video_audio(mouth_path)
        mouth_mid = mouth_raw[mouth_raw.shape[0] // 2].float() / 255.0
        axes[r, 1].imshow(mouth_mid.permute(1, 2, 0).numpy())

        for c in range(2):
            axes[r, c].axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    out_path = os.path.join(OUT_DIR, 'roi_grid_2x2.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
