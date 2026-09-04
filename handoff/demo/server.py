"""Demo server: serves the static site (like serve.py) AND a live-recording
endpoint that crops a freshly recorded webcam clip to the same mouth-ROI format
the models were trained on, then runs all three models on it.

Needs the full model code + checkpoints in ../models/ (already shipped in this
handoff) plus mediapipe + flask (see ../requirements.txt). No GLips dataset
access needed -- unlike build_demo_data.py, this runs standalone from inside the
handoff folder.

Usage:  python server.py [port]   (default 8000)
"""
import os
import sys
import uuid
import wave
import tempfile
import subprocess

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(HERE, 'clips')
RECORDED_DIR = os.path.join(CLIPS_DIR, 'recorded')
os.makedirs(RECORDED_DIR, exist_ok=True)

sys.path.insert(0, HERE)
import live_infer  # noqa: E402  (loads models -- slow import, do it once at startup)

app = Flask(__name__, static_folder=None)


@app.after_request
def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp


@app.route('/')
def index():
    return send_from_directory(HERE, 'index.html')


@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(HERE, path)


@app.route('/api/classes')
def classes():
    return jsonify(live_infer.CLASSES)


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _to_mp4(src_path, dst_path):
    """Re-encode whatever the browser sent (webm/vp8/opus) to a standard h264/aac
    mp4 so cv2.VideoCapture and whisper's ffmpeg call both read it reliably."""
    cmd = [_ffmpeg(), '-y', '-hide_banner', '-loglevel', 'error', '-i', src_path,
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', dst_path]
    subprocess.run(cmd, check=True)


def _extract_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _encode_cropped_with_audio(mouth_crops, audio_path, out_path, size, fps=25.0):
    """Encode the BGR uint8 mouth-crop frames straight through ffmpeg (piped as
    rawvideo) and mux in the given audio in one pass, producing a standard h264/
    aac mp4. Deliberately avoids cv2.VideoWriter: its 'mp4v' fourcc is flaky on
    Windows -- it can silently produce a zero-byte/corrupt file (isOpened() goes
    unchecked), which then either fails the old separate mux step or, worse, gets
    copied straight to the browser as an unplayable "result" clip."""
    cmd = [_ffmpeg(), '-y', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{size}x{size}', '-r', str(fps),
           '-i', '-', '-i', audio_path,
           '-map', '0:v:0', '-map', '1:a?',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
           '-shortest', '-movflags', '+faststart', out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in mouth_crops:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def _write_wav_int16(path, wav_np, sr=16000):
    pcm = (np.clip(wav_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


@app.route('/api/record', methods=['POST'])
def record():
    if 'video' not in request.files:
        return jsonify({'error': 'kein Video im Request'}), 400
    target = request.form.get('target') or None
    condition = request.form.get('condition') or 'clean'

    with tempfile.TemporaryDirectory() as tmp:
        raw_path = os.path.join(tmp, 'raw.webm')
        request.files['video'].save(raw_path)

        mp4_path = os.path.join(tmp, 'src.mp4')
        try:
            _to_mp4(raw_path, mp4_path)
        except subprocess.CalledProcessError:
            return jsonify({'error': 'Video konnte nicht dekodiert werden'}), 400

        frames = _extract_frames(mp4_path)
        if len(frames) < 3:
            return jsonify({'error': 'Zu wenige Frames aufgenommen -- nochmal versuchen'}), 400

        mouth_crops, detect_rate = live_infer.crop_mouth_frames(frames)

        import whisper
        try:
            wav = whisper.load_audio(mp4_path)
        except Exception:
            wav = None

        preds, noisy_wav = live_infer.predict(
            mouth_crops, wav if wav is not None else np.zeros(16000, dtype=np.float32),
            target, condition)

        # encode the cropped mouth-ROI frames + exactly the (possibly noised)
        # audio the models actually heard into one playable mp4, so preview
        # matches the result.
        heard_wav_path = os.path.join(tmp, 'heard.wav')
        _write_wav_int16(heard_wav_path, noisy_wav)

        out_name = f"{uuid.uuid4().hex[:12]}.mp4"
        out_path = os.path.join(RECORDED_DIR, out_name)
        try:
            _encode_cropped_with_audio(mouth_crops, heard_wav_path, out_path, live_infer.OUT_SIZE)
        except subprocess.CalledProcessError:
            return jsonify({'error': 'Clip konnte nicht kodiert werden -- nochmal versuchen'}), 500
        clip_file = f"recorded/{out_name}"

    return jsonify({
        'clip_file': clip_file,
        'target': target,
        'condition': condition,
        'detect_rate': round(detect_rate, 3),
        **preds,
    })


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app.run(host='127.0.0.1', port=port, debug=False)
