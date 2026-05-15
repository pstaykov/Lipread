import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from pathlib import Path


def parse_time(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000


def parse_srt(path):
    text = open(path).read()
    pattern = r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+'
    blocks = re.split(r'\n\n+', text.strip())
    words = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        times = lines[1].split(' --> ')
        start, end = parse_time(times[0].strip()), parse_time(times[1].strip())
        for word in ' '.join(lines[2:]).strip().split():
            words.append({"start": start, "end": end, "word": word})
    return words


def build_labels(words, num_frames, fps):
    labels = np.zeros(num_frames, dtype=np.float32)
    for w in words:
        s = int(w["start"] * fps)
        e = min(int(w["end"] * fps), num_frames)
        labels[s:e] = 1.0
    return labels


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames), fps


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96, 96)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class WordBoundaryDataset(Dataset):
    def __init__(self, video_paths, srt_paths, clip_len=75, stride=25):
        self.clips = []
        self.labels = []

        for vp, sp in zip(video_paths, srt_paths):
            frames, fps = load_video_frames(vp)
            words = parse_srt(sp)
            label_seq = build_labels(words, len(frames), fps)

            for start in range(0, len(frames) - clip_len, stride):
                clip_frames = frames[start:start + clip_len]
                clip_labels = label_seq[start:start + clip_len]

                tensors = torch.stack([TRANSFORM(f) for f in clip_frames])
                self.clips.append(tensors)
                self.labels.append(torch.tensor(clip_labels))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        return self.clips[idx], self.labels[idx]