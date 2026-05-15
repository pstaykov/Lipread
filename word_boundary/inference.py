import torch
import numpy as np
import cv2
from torchvision import transforms
from model import WordBoundaryDetector


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((96, 96)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict(video_path, checkpoint, clip_len=75, stride=25,
            threshold=0.5, boundary_threshold=0.3, device="cuda"):
    model = WordBoundaryDetector().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    all_probs = np.zeros(len(frames))
    counts = np.zeros(len(frames))

    for start in range(0, len(frames) - clip_len, stride):
        clip = torch.stack([TRANSFORM(f) for f in frames[start:start + clip_len]])
        clip = clip.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(clip)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        all_probs[start:start + clip_len] += probs
        counts[start:start + clip_len] += 1

    counts = np.maximum(counts, 1)
    probs = all_probs / counts

    speaking = probs > threshold
    diff = np.abs(np.diff(probs))
    boundaries = np.where(diff > boundary_threshold)[0]

    word_intervals = []
    for b in boundaries:
        t = b / fps
        word_intervals.append({"frame": int(b), "time_sec": round(t, 3)})

    return {"speaking_prob": probs, "speaking": speaking,
            "boundaries": word_intervals, "fps": fps}