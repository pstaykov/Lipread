import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'lipreading'))
from dataset import GLipsFullClipDataset, GLips15Dataset, _load_video_audio  # noqa: F401

try:
    import whisper as whisper_lib
except ImportError:
    raise ImportError("openai-whisper not found. Install with:  pip install openai-whisper")


class GLipsAVDataset(GLipsFullClipDataset):
    """Extends GLipsFullClipDataset to also return a Whisper mel spectrogram.

    Returns (video (C,T,H,W), mel (80,3000), label).
    Accepts an optional `classes` list to restrict to a subset of words.
    """
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        video, audio_wav = _load_video_audio(video_path)
        if audio_wav is None:
            audio_wav = np.zeros(16000 * 30, dtype=np.float32)
        video = self._process_video(video)
        mel = whisper_lib.log_mel_spectrogram(whisper_lib.pad_or_trim(audio_wav))
        return video, mel, label
