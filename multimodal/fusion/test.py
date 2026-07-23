import os
import random
import torch
import tkinter as tk
from PIL import Image, ImageTk

from train import (
    GLipsNet, MultimodalGLipsDataset, VideoAugment,
    WhisperExtractor, _strip_orig_mod,
    WHISPER_MODEL_NAME, AUDIO_T, AUDIO_DIM,
    _ensure_ffmpeg,
)

_ensure_ffmpeg()

ROOT_DIR = '../../lipreading/GLips/lipread_files'
CHECKPOINT = './models/checkpoints/best_model.pth'
NUM_SAMPLES = 5
NUM_FRAMES = 25


def load_raw_frames(video_path):
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
        return frames
    except Exception:
        return []


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Whisper only needed if audio caches are missing
    extractor = WhisperExtractor(WHISPER_MODEL_NAME, device=device)

    val_tf = VideoAugment(crop_size=88, resize_size=96, is_train=False)
    dataset = MultimodalGLipsDataset(ROOT_DIR, split='test', num_frames=NUM_FRAMES,
                                     transform=val_tf, extractor=extractor)
    if len(dataset) == 0:
        dataset = MultimodalGLipsDataset(ROOT_DIR, split='validation', num_frames=NUM_FRAMES,
                                         transform=val_tf, extractor=extractor)

    classes = dataset.classes
    print(f"Loaded {len(dataset)} samples, {len(classes)} classes.")

    model = GLipsNet(num_classes=len(classes), use_audio=True)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(_strip_orig_mod(state))
    model.to(device)
    model.eval()

    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    samples = []
    for idx in indices:
        video, audio, label = dataset[idx]
        video_path, _ = dataset.samples[idx]
        samples.append({
            'video': video,
            'audio': audio,
            'label': label,
            'class_name': classes[label],
            'raw_frames': load_raw_frames(video_path),
        })

    with torch.no_grad():
        for s in samples:
            logits = model(
                s['video'].unsqueeze(0).to(device),
                s['audio'].unsqueeze(0).to(device),
            )
            top3 = logits.topk(3, dim=1)
            s['predicted'] = classes[top3.indices[0, 0].item()]
            s['top3'] = [(classes[top3.indices[0, k].item()],
                          top3.values[0, k].item()) for k in range(3)]

    # --- GUI ---
    root = tk.Tk()
    root.title('GLips Multimodal — Test')
    root.resizable(False, False)
    root.configure(bg='#1e1e2e')

    current = [0]
    frame_idx = [0]
    after_id = [None]

    CANVAS_W, CANVAS_H = 320, 220

    canvas = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg='black', highlightthickness=0)
    canvas.pack(padx=20, pady=(20, 8))

    pred_var = tk.StringVar()
    pred_label = tk.Label(root, textvariable=pred_var, font=('Helvetica', 15, 'bold'),
                          bg='#1e1e2e', fg='#cba6f7', justify='center')
    pred_label.pack(pady=(4, 0))

    gt_var = tk.StringVar()
    tk.Label(root, textvariable=gt_var, font=('Helvetica', 12),
             bg='#1e1e2e', fg='#a6e3a1', justify='center').pack(pady=(2, 0))

    top3_var = tk.StringVar()
    tk.Label(root, textvariable=top3_var, font=('Helvetica', 10),
             bg='#1e1e2e', fg='#6c7086', justify='center').pack(pady=(2, 4))

    counter_var = tk.StringVar()
    tk.Label(root, textvariable=counter_var, font=('Helvetica', 10),
             bg='#1e1e2e', fg='#585b70').pack()

    btn_frame = tk.Frame(root, bg='#1e1e2e')
    btn_frame.pack(pady=14)

    def show_sample(i):
        frame_idx[0] = 0
        s = samples[i]
        correct = s['predicted'] == s['class_name']
        pred_var.set(f"Predicted:  {s['predicted']}  {'✓' if correct else '✗'}")
        pred_label.config(fg='#a6e3a1' if correct else '#f38ba8')
        gt_var.set(f"Ground truth:  {s['class_name']}")
        top3_var.set('Top-3:  ' + '  |  '.join(f"{w} ({v:.1f})" for w, v in s['top3']))
        counter_var.set(f"Sample {i + 1} / {len(samples)}")
        _animate()

    def _animate():
        if after_id[0] is not None:
            root.after_cancel(after_id[0])
        frames = samples[current[0]]['raw_frames']
        if not frames:
            canvas.create_text(CANVAS_W // 2, CANVAS_H // 2, text='No frames',
                               fill='white', font=('Helvetica', 14))
            return
        fi = frame_idx[0] % len(frames)
        img = Image.fromarray(frames[fi]).resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        canvas.image = photo
        canvas.create_image(0, 0, anchor='nw', image=photo)
        frame_idx[0] = fi + 1
        after_id[0] = root.after(40, _animate)  # ~25 fps

    def prev_sample():
        if after_id[0]:
            root.after_cancel(after_id[0])
        current[0] = (current[0] - 1) % len(samples)
        show_sample(current[0])

    def next_sample():
        if after_id[0]:
            root.after_cancel(after_id[0])
        current[0] = (current[0] + 1) % len(samples)
        show_sample(current[0])

    btn_style = dict(bg='#313244', fg='white', relief='flat',
                     activebackground='#45475a', activeforeground='white',
                     padx=20, pady=8, font=('Helvetica', 12))
    tk.Button(btn_frame, text='◀  Prev', command=prev_sample, **btn_style).pack(side='left', padx=10)
    tk.Button(btn_frame, text='Next  ▶', command=next_sample, **btn_style).pack(side='left', padx=10)

    show_sample(0)
    root.mainloop()


if __name__ == '__main__':
    main()
