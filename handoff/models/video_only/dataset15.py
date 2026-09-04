"""Canonical GLips-15 benchmark config: the 15 word classes and dataset builders.

Single source of truth so the class list can never silently diverge between the
Transformer and MS-TCN trainers (they previously each held their own copy).

The 15 classes are **Ameer et al.'s** set, and we evaluate on the **stock GLips
folder split** (the fixed 400 train / 50 val / 50 test clips the corpus ships
with, ``group_split=False``) so our numbers are directly comparable to theirs.
No source-disjoint re-partitioning is applied: the published train/val/test
folders are treated as authoritative.
"""
import os

from dataset import GLipsFullClipDataset, VideoAugment

# Ameer et al.'s 15 GLips word classes (as read off their confusion matrix).
CLASSES = sorted([
    'aber', 'aufgaben', 'bleibt', 'darüber', 'digitalisierung',
    'einmal', 'finden', 'gegen', 'geworden', 'herrn',
    'investitionen', 'kommunen', 'linke', 'möglichkeit', 'passiert',
])

_LIPREAD_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_ROOT = os.path.join(_LIPREAD_DIR, 'GLips_mouth', 'lipread_files')   # mouth-ROI crops
FULL_ROOT = os.path.join(_LIPREAD_DIR, 'GLips', 'lipread_files')        # uncropped full-face

NUM_FRAMES = 25
# Stock published split — compare head-to-head with Ameer et al.
GROUP_SPLIT = False

# Ameer et al.'s exact input pipeline (for the 15-word head-to-head runs).
AMEER_FRAMES = 16     # they select 16 frames per clip
AMEER_SIZE = 128      # resize to 128x128, no crop


def stock_complete_classes(root_dir=ROI_ROOT):
    """Sorted class names that have a complete stock split (>=1 clip in BOTH a train
    and a val/validation folder). Safety filter so train and val are drawn from an
    identical, aligned class set even if a class is ever incomplete on disk.
    """
    def _has_mp4(*parts):
        d = os.path.join(root_dir, *parts)
        return os.path.isdir(d) and any(f.endswith('.mp4') for f in os.listdir(d))

    out = []
    for c in sorted(os.listdir(root_dir)):
        if not os.path.isdir(os.path.join(root_dir, c)):
            continue
        if _has_mp4(c, 'train') and (_has_mp4(c, 'val') or _has_mp4(c, 'validation')):
            out.append(c)
    return out


def make_15_datasets(root_dir=ROI_ROOT, num_frames=NUM_FRAMES, group_split=GROUP_SPLIT):
    """(train_dataset, val_dataset) for the 15-class task under the standard recipe:
    grayscale + random-erasing + temporal speed-perturbation on train, deterministic
    resize/centre-crop on val. Point ``root_dir`` at FULL_ROOT for the no-mouth-ROI
    ablation; everything else is held identical for a fair comparison."""
    train_transform = VideoAugment(crop_size=88, resize_size=96, is_train=True,
                                   grayscale_p=0.2, random_erase=0.25)
    val_transform = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    train_dataset = GLipsFullClipDataset(root_dir, split='train', num_frames=num_frames,
                                         transform=train_transform, classes=CLASSES,
                                         temporal_jitter=True, group_split=group_split)
    val_dataset = GLipsFullClipDataset(root_dir, split='validation', num_frames=num_frames,
                                       transform=val_transform, classes=CLASSES,
                                       group_split=group_split)
    return train_dataset, val_dataset


def make_15_datasets_ameer():
    """Ameer et al.'s EXACT input pipeline for the 15-word head-to-head runs.

    Uncropped full-face frames (FULL_ROOT, no face detection / no mouth-ROI crop),
    resize 128x128 with NO random crop, 16 frames/clip, min-max [0,1] normalization,
    and horizontal-flip-only image augmentation (no affine, photometric, grayscale,
    random-erase, time-mask, or temporal jitter). The same-class interpolation and
    the plain recipe (no EMA / no label smoothing) are applied by the trainer, not
    here. Only the augmentation/preprocessing matches Ameer — the model stays ours
    (GLipsNet / MS-TCN), which is the whole point of the comparison.
    """
    train_transform = VideoAugment(
        crop_size=AMEER_SIZE, resize_size=AMEER_SIZE, is_train=True,
        brightness=0.0, contrast=0.0, grayscale_p=0.0, random_erase=0.0,
        rotation_deg=0.0, scale_jitter=0.0, translate_frac=0.0,
        time_mask_max=0, normalize='minmax')                 # -> flip-only remains active
    val_transform = VideoAugment(
        crop_size=AMEER_SIZE, resize_size=AMEER_SIZE, is_train=False, normalize='minmax')
    train_dataset = GLipsFullClipDataset(FULL_ROOT, split='train', num_frames=AMEER_FRAMES,
                                         transform=train_transform, classes=CLASSES,
                                         temporal_jitter=False, group_split=False)
    val_dataset = GLipsFullClipDataset(FULL_ROOT, split='validation', num_frames=AMEER_FRAMES,
                                       transform=val_transform, classes=CLASSES,
                                       group_split=False)
    return train_dataset, val_dataset
