import os
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import torchvision.io as io
except RuntimeError:
    io = None


def _load_video_audio(video_path):
    """Return (video TCHW tensor, audio_wav float32 numpy 16 kHz or None).

    Tries torchvision.io first (captures both streams in one call), then
    imageio and cv2 as video-only fallbacks.  Audio fallback via torchaudio.
    """
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
    """Video-only dataset over the GLips folder layout.

    root_dir/
        <class>/
            train/  validation/  test/  *.mp4

    If `classes` is provided, only those class names are loaded (and indexed
    in the given order).  Otherwise all subdirectories are discovered.
    """
    def __init__(self, root_dir, split='train', transform=None, num_frames=25, classes=None):
        self.transform = transform
        self.num_frames = num_frames
        self.samples = []

        if classes is None:
            classes = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls_name in self.classes:
            split_folder = os.path.join(root_dir, cls_name, split)
            if not os.path.exists(split_folder):
                for alt in ['train', 'validation', 'val', 'test']:
                    alt_folder = os.path.join(root_dir, cls_name, alt)
                    if os.path.exists(alt_folder):
                        split_folder = alt_folder
                        break
            if os.path.exists(split_folder):
                for f in os.listdir(split_folder):
                    if f.endswith('.mp4'):
                        self.samples.append(
                            (os.path.join(split_folder, f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def _process_video(self, video):
        """Normalize, temporally sample, augment, and permute to (C, T, H, W)."""
        video = video.float() / 255.0
        T = video.size(0)
        if T > self.num_frames:
            indices = np.linspace(0, T - 1, num=self.num_frames).astype(int)
            video = video[indices]
        elif T < self.num_frames:
            pad = self.num_frames - T
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
