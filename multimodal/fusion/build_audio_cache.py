"""Decode every clip's .m4a into a fixed-length waveform memmap cache for the trainer."""
import os
import glob
import json
import importlib.util
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
from tqdm import tqdm

_spec = importlib.util.spec_from_file_location(
    "mmtrain", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"))
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

ROOT_DIR = '../../lipreading/GLips_mouth/lipread_files'
AUDIO_ROOT = '../../lipreading/GLips/lipread_files'
CACHE_DIR = './cache/audio_cache'
AUDIO_SAMPLES = m.AUDIO_SAMPLES
NUM_THREADS = 12   # pyav decodes in-process and releases the GIL, so threads scale well


def load_audio_pyav(path, sr=16000):
    """Decode to 16 kHz mono float32 in-process (bit-identical to whisper.load_audio, ~4x faster)."""
    container = av.open(path)
    resampler = av.AudioResampler(format='s16', layout='mono', rate=sr)
    chunks = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    for rf in (resampler.resample(None) or []):   # flush
        chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def gather_items():
    """Unique {key: audio_path} for every clip across all class/split folders."""
    root_norm = os.path.normpath(ROOT_DIR)
    items = {}
    for video_path in glob.glob(os.path.join(ROOT_DIR, '*', '*', '*.mp4')):
        key = m._audio_key(video_path, root_norm)
        if key not in items:
            items[key] = os.path.join(AUDIO_ROOT, key + '.m4a')
    return list(items.items())


def decode(idx_key_path):
    idx, (key, apath) = idx_key_path
    try:
        wav = m._fix_wav(load_audio_pyav(apath)).numpy().astype(np.float16)
    except Exception:
        wav = np.zeros(AUDIO_SAMPLES, dtype=np.float16)
    return idx, key, wav


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    items = gather_items()
    n = len(items)
    print(f"{n} unique clips -> {n * AUDIO_SAMPLES * 2 / 1e9:.1f} GB cache at {CACHE_DIR}")

    mm = np.memmap(os.path.join(CACHE_DIR, 'waveforms.dat'),
                   dtype=np.float16, mode='w+', shape=(n, AUDIO_SAMPLES))
    rows = {}
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as ex:
        for idx, key, wav in tqdm(ex.map(decode, enumerate(items)),
                                  total=n, desc='decoding', unit='clip'):
            mm[idx] = wav
            rows[key] = idx
    mm.flush()
    del mm

    with open(os.path.join(CACHE_DIR, 'index.json'), 'w') as f:
        json.dump({'n': n, 'audio_samples': AUDIO_SAMPLES, 'rows': rows}, f)
    print(f"Done. {n} waveforms cached. index.json written.")


if __name__ == '__main__':
    main()
