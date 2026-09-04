import os
import re
import hashlib
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

try:
    import torchvision.io as io
except RuntimeError:
    io = None


class VideoAugment:
    """Spatial + temporal augmentation, one random draw per clip applied identically to every frame."""
    def __init__(self, crop_size=88, resize_size=96, is_train=True, time_mask_max=4,
                 max_time_masks=2, rotation_deg=10.0, scale_jitter=0.1,
                 translate_frac=0.06, brightness=0.2, contrast=0.2,
                 grayscale_p=0.0, random_erase=0.0, erase_scale=(0.02, 0.2),
                 normalize='imagenet'):
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.is_train = is_train
        self.time_mask_max = time_mask_max
        self.max_time_masks = max_time_masks
        self.rotation_deg = rotation_deg
        self.scale_jitter = scale_jitter
        self.translate_frac = translate_frac
        self.brightness = brightness
        self.contrast = contrast
        self.grayscale_p = grayscale_p    # prob of dropping colour for the whole clip
        self.random_erase = random_erase  # prob of zeroing one rectangle (cutout-style)
        self.erase_scale = erase_scale
        self.normalize = normalize        # 'imagenet' (default) or 'minmax' (Ameer et al.'s scheme)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    @staticmethod
    def _sym(mag):
        """Uniform in [-mag, +mag]."""
        return (torch.rand(1).item() * 2 - 1) * mag

    def __call__(self, video):
        T = video.shape[0]

        if self.resize_size is not None:
            video = torch.stack([
                TF.resize(video[t], [self.resize_size, self.resize_size], antialias=True)
                for t in range(T)
            ])

        if self.is_train:
            if self.brightness > 0:
                video = TF.adjust_brightness(video, 1.0 + self._sym(self.brightness))
            if self.contrast > 0:
                video = TF.adjust_contrast(video, 1.0 + self._sym(self.contrast))
            if self.grayscale_p > 0 and torch.rand(1).item() < self.grayscale_p:
                video = TF.rgb_to_grayscale(video, num_output_channels=3)
            video = video.clamp(0, 1)

            # applied before cropping so the border padding is mostly cropped away afterwards
            if self.rotation_deg > 0 or self.scale_jitter > 0 or self.translate_frac > 0:
                H0, W0 = video.shape[2], video.shape[3]
                video = TF.affine(
                    video,
                    angle=self._sym(self.rotation_deg),
                    translate=[int(round(self._sym(self.translate_frac * W0))),
                               int(round(self._sym(self.translate_frac * H0)))],
                    scale=1.0 + self._sym(self.scale_jitter),
                    shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BILINEAR,
                )

        H, W = video.shape[2], video.shape[3]
        cs = self.crop_size
        if H >= cs and W >= cs:
            if self.is_train:
                top = torch.randint(0, H - cs + 1, (1,)).item()
                left = torch.randint(0, W - cs + 1, (1,)).item()
            else:
                top = (H - cs) // 2
                left = (W - cs) // 2
            video = video[:, :, top:top + cs, left:left + cs]

        if self.is_train:
            if torch.rand(1).item() < 0.5:
                video = torch.flip(video, dims=[3])

            if self.time_mask_max > 0 and T > 1:
                video = video.clone()
                for _ in range(torch.randint(1, self.max_time_masks + 1, (1,)).item()):
                    n = torch.randint(1, self.time_mask_max + 1, (1,)).item()
                    start = torch.randint(0, max(1, T - n + 1), (1,)).item()
                    video[start:start + n] = 0.0

            if self.random_erase > 0 and torch.rand(1).item() < self.random_erase:
                video = self._erase(video)

        if self.normalize == 'minmax':
            lo, hi = video.amin(), video.amax()
            video = (video - lo) / (hi - lo + 1e-6)
        else:
            video = (video - self.mean) / self.std
        return video

    def _erase(self, video):
        """Zero one rectangle, identical box across all T frames."""
        T, C, H, W = video.shape
        area = H * W
        for _ in range(10):
            er = area * random.uniform(*self.erase_scale)
            ar = random.uniform(0.3, 3.3)
            h = int(round((er * ar) ** 0.5))
            w = int(round((er / ar) ** 0.5))
            if 0 < h < H and 0 < w < W:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                video = video.clone()
                video[:, :, i:i + h, j:j + w] = random.random()  # fill with a flat grey
                break
        return video


def _load_video_audio(video_path):
    """Return (video TCHW tensor, audio_wav float32 numpy 16 kHz or None); torchvision.io then imageio/cv2/torchaudio fallbacks."""
    video = None
    audio_wav = None

    if io is not None and hasattr(io, 'read_video'):
        try:
            video, audio, info = io.read_video(video_path, pts_unit='sec', output_format='TCHW')
            if audio.numel() > 0:
                sr = int(info.get('audio_fps', 16000))
                wav = audio.float().mean(dim=0)
                if sr != 16000:
                    try:
                        import torchaudio
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    except ImportError:
                        pass
                audio_wav = wav.numpy()
        except Exception:
            video = None
            audio_wav = None

    if video is None:
        try:
            import imageio.v2 as imageio
        except Exception:
            try:
                import imageio
            except Exception:
                imageio = None
        if imageio is not None:
            frames = []
            try:
                reader = imageio.get_reader(video_path, 'ffmpeg')
                for frame in reader:
                    frames.append(frame)
                reader.close()
                video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            except Exception:
                video = None

    if video is None:
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames:
                raise RuntimeError(f"No frames read from {video_path}")
            video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
        except Exception as e:
            raise RuntimeError(f"Could not read video. Install imageio or opencv. Error: {e}")

    if audio_wav is None:
        try:
            import torchaudio
            wav, sr = torchaudio.load(video_path)
            wav = wav.float().mean(dim=0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            audio_wav = wav.numpy()
        except Exception:
            audio_wav = None  # callers that need audio should substitute zeros

    return video, audio_wav


class GLipsFullClipDataset(Dataset):
    """Video-only dataset over the GLips folder layout: root_dir/<class>/{train,val,test}/*.mp4."""

    # a split must NEVER resolve to a different split's folder, or it silently leaks data and fakes metrics
    SPLIT_ALIASES = {
        'train': ('train',),
        'validation': ('val', 'validation'),
        'val': ('val', 'validation'),
        'test': ('test',),
    }

    # filenames are "<word>_<SOURCE>-<CLIP>.mp4"; group_split (default) re-partitions by SOURCE
    # so a broadcast never spans splits (the stock per-clip folders leak ~66% of val sources into train)
    _SOURCE_RE = re.compile(r'_(\d+)-\d+\.mp4$')

    @classmethod
    def _source_id(cls, filename):
        m = cls._SOURCE_RE.search(filename)
        return m.group(1) if m else filename  # fall back to filename = its own group

    @staticmethod
    def _grouped_split(source_id, val_frac, test_frac):
        """Deterministically hash a source-ID to 'train'/'val'/'test' (stable across runs, unlike Python's salted hash)."""
        h = int(hashlib.md5(source_id.encode()).hexdigest(), 16) % 10_000 / 10_000.0
        if h < test_frac:
            return 'test'
        if h < test_frac + val_frac:
            return 'val'
        return 'train'

    def __init__(self, root_dir, split='train', transform=None, num_frames=25, classes=None,
                 temporal_jitter=False, jitter_speed=(0.8, 1.2),
                 group_split=True, group_val_frac=0.1, group_test_frac=0.1,
                 require_all_classes=True):
        self.transform = transform
        self.num_frames = num_frames
        self.temporal_jitter = temporal_jitter  # train only: sample from a speed-warped sub-window
        self.jitter_speed = jitter_speed
        self.samples = []

        if classes is None:
            classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if group_split:
            self._build_grouped(root_dir, split, group_val_frac, group_test_frac)
            return

        candidates = self.SPLIT_ALIASES.get(split, (split,))
        missing = []
        for cls_name in self.classes:
            split_folder = next(
                (os.path.join(root_dir, cls_name, a) for a in candidates
                 if os.path.isdir(os.path.join(root_dir, cls_name, a))),
                None)
            if split_folder is None:
                missing.append(cls_name)
                continue
            for f in os.listdir(split_folder):
                if f.endswith('.mp4'):
                    self.samples.append(
                        (os.path.join(split_folder, f), self.class_to_idx[cls_name]))

        if missing and require_all_classes:
            raise FileNotFoundError(
                f"Split '{split}' (folders {candidates}) not found for "
                f"{len(missing)}/{len(self.classes)} classes, e.g. "
                f"{missing[:3]}. Refusing to silently fall back to another "
                f"split — that would leak data and corrupt metrics. "
                f"Pass require_all_classes=False to allow a class to contribute "
                f"zero samples to this split (its label index is still reserved).")
        if missing:
            print(f"[dataset] '{split}': {len(missing)} of {len(self.classes)} classes have no "
                  f"such folder and contribute 0 samples (e.g. {missing[:3]}).")

    def _build_grouped(self, root_dir, split, val_frac, test_frac):
        """Source-disjoint split: pool every clip across the stock folders, keep only those whose SOURCE hashes to `split`."""
        want = 'val' if split in ('val', 'validation') else split
        if want not in ('train', 'val', 'test'):
            raise ValueError(f"group_split supports train/val/test, got '{split}'")
        for cls_name in self.classes:
            for sub in ('train', 'val', 'validation', 'test'):
                d = os.path.join(root_dir, cls_name, sub)
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    if not f.endswith('.mp4'):
                        continue
                    if self._grouped_split(self._source_id(f), val_frac, test_frac) == want:
                        self.samples.append((os.path.join(d, f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def _process_video(self, video):
        """Normalize, temporally sample, augment, and permute to (C, T, H, W)."""
        video = video.float() / 255.0
        T = video.size(0)
        n = self.num_frames
        if self.temporal_jitter and T > n:
            speed = random.uniform(*self.jitter_speed)  # <1 squeezes (slower motion), >1 spreads out
            L = max(2, min(T, int(round((n - 1) * speed)) + 1))
            start = random.randint(0, T - L)
            indices = np.linspace(start, start + L - 1, num=n).astype(int)
            video = video[indices]
        elif T > n:
            indices = np.linspace(0, T - 1, num=n).astype(int)
            video = video[indices]
        elif T < n:
            pad = n - T
            video = torch.cat([video, video[-1:].repeat(pad, 1, 1, 1)], dim=0)
        if self.transform:
            video = self.transform(video)
        return video.permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, _ = _load_video_audio(video_path)
        return self._process_video(video), label


class GLips15Dataset(GLipsFullClipDataset):
    """Restricts the dataset to the first `num_classes` alphabetically-sorted classes."""
    def __init__(self, root_dir, split='train', transform=None, num_frames=25, num_classes=15):
        all_classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        super().__init__(root_dir, split, transform, num_frames,
                         classes=all_classes[:num_classes])
