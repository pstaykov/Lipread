"""Canonical GLips-15 benchmark config: Ameer et al.'s 15 word classes, evaluated on the stock (non-grouped) GLips split."""
import os

from dataset import GLipsFullClipDataset, VideoAugment

CLASSES = sorted([  # Ameer et al.'s 15 GLips word classes (read off their confusion matrix)
    'aber', 'aufgaben', 'bleibt', 'darüber', 'digitalisierung',
    'einmal', 'finden', 'gegen', 'geworden', 'herrn',
    'investitionen', 'kommunen', 'linke', 'möglichkeit', 'passiert',
])

_LIPREAD_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_ROOT = os.path.join(_LIPREAD_DIR, 'GLips_mouth', 'lipread_files')   # mouth-ROI crops
FULL_ROOT = os.path.join(_LIPREAD_DIR, 'GLips', 'lipread_files')        # uncropped full-face

NUM_FRAMES = 25
GROUP_SPLIT = False   # stock published split, for head-to-head comparison with Ameer et al.

AMEER_FRAMES = 16     # they select 16 frames per clip
AMEER_SIZE = 128      # resize to 128x128, no crop


def stock_complete_classes(root_dir=ROI_ROOT):
    """Sorted class names with a complete stock split (>=1 clip in both a train and a val/validation folder)."""
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
    """(train_dataset, val_dataset) for the 15-class task: standard train augmentation, deterministic val."""
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
    """Ameer et al.'s exact input pipeline (uncropped faces, 128x128 no-crop, minmax norm, flip-only aug); model stays ours."""
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
