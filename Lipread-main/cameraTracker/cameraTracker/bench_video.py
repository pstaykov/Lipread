import subprocess
import sys
from pathlib import Path
import glob
import time

# download clip using yt-dlp
url = 'https://www.youtube.com/watch?v=aqz-KE-bpKQ'
out_pattern = 'bench_clip.%(ext)s'
cmd = [sys.executable, '-m', 'yt_dlp', '-f', 'best', '--download-sections', '*00:00:00-00:00:10', '-o', out_pattern, url]
print('Running:', ' '.join(cmd))
res = subprocess.run(cmd)
if res.returncode != 0:
    print('yt-dlp failed with code', res.returncode)
    sys.exit(res.returncode)

# find downloaded file
files = glob.glob('bench_clip.*')
if not files:
    print('No downloaded clip found')
    sys.exit(1)
video_path = files[0]
print('Downloaded:', video_path)

# run benchmark using camera_preview's YuNetDetector
import cv2
import numpy as np
from camera_preview import YuNetDetector, ensure_yunet_model

p = ensure_yunet_model()
print('Model path:', p)
try:
    det_cpu = YuNetDetector(p, min_detection_confidence=0.01, prefer_gpu=False)
    det_gpu = YuNetDetector(p, min_detection_confidence=0.01, prefer_gpu=True)
except Exception as e:
    print('Detector init error:', e)
    sys.exit(1)

print('CPU providers:', det_cpu.session.get_providers())
print('GPU providers:', det_gpu.session.get_providers())

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Could not open video', video_path)
    sys.exit(1)

# warmup frames
for _ in range(5):
    ok, frame = cap.read()
    if not ok:
        break
    det_cpu.detect(frame)

# timed run up to 150 frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0
times = []
while frame_count < 150:
    ok, frame = cap.read()
    if not ok:
        break
    t0 = time.perf_counter(); det_cpu.detect(frame); t1 = time.perf_counter()
    times.append((t1 - t0) * 1000.0)
    frame_count += 1

cap.release()

if times:
    print(f'Processed {len(times)} frames')
    print('avg CPU ms/frame:', sum(times)/len(times))
else:
    print('No frames processed')

# GPU run if different session
if det_gpu is not det_cpu:
    cap = cv2.VideoCapture(video_path)
    for _ in range(5):
        ok, frame = cap.read()
        if not ok:
            break
        det_gpu.detect(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    times_g = []
    fc = 0
    while fc < 150:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter(); det_gpu.detect(frame); t1 = time.perf_counter()
        times_g.append((t1 - t0) * 1000.0)
        fc += 1
    cap.release()
    if times_g:
        print('avg GPU ms/frame:', sum(times_g)/len(times_g))
    else:
        print('No GPU frames processed')
else:
    print('GPU provider equals CPU provider; GPU run skipped')
