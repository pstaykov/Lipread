"""Audio-only ASR baseline: transcribe each clip with Whisper and map the first word to the nearest class.

    SPLIT=val,test  LIMIT=0  python eval_asr_baseline.py
Writes asr_baseline.csv with exact-match and nearest-class top1 per split.
"""
import os
import sys
import csv
import json
import difflib
import importlib.util
import numpy as np
import torch
import whisper
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
if 'mmtrain' not in sys.modules:
    spec = importlib.util.spec_from_file_location('mmtrain', os.path.join(HERE, 'train.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['mmtrain'] = m
    spec.loader.exec_module(m)
else:
    m = sys.modules['mmtrain']

AUD_CACHE = os.path.join(HERE, 'cache', 'audio_cache')
ROOT = os.path.join(HERE, '..', 'lipreading', 'GLips_mouth', 'lipread_files')
EXCLUDED = ('hier', 'soll')
SPLITS = os.environ.get('SPLIT', 'val,test').split(',')
LIMIT = int(os.environ.get('LIMIT', '0'))
BATCH = 64


def norm(s):
    """Lowercase, letters only (umlauts kept)."""
    return ''.join(ch for ch in s.lower() if ch.isalpha())


def main():
    device = torch.device('cuda')
    classes = sorted(d for d in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, d)) and d not in EXCLUDED)
    assert len(classes) == 498
    classes_norm = [norm(c) for c in classes]
    norm_to_class = {cn: c for cn, c in zip(classes_norm, classes)}

    with open(os.path.join(AUD_CACHE, 'index.json')) as f:
        aidx = json.load(f)
    wav = np.memmap(os.path.join(AUD_CACHE, 'waveforms.dat'), dtype=np.float16,
                    mode='r', shape=(aidx['n'], aidx['audio_samples']))

    model = whisper.load_model('base', device=device)
    model.eval()
    n_mels = model.dims.n_mels
    options = whisper.DecodingOptions(language='de', task='transcribe',
                                      without_timestamps=True, fp16=True, sample_len=12)

    results_rows = []
    for split in SPLITS:
        split = split.strip()
        items = [(k, r) for k, r in aidx['rows'].items() if k.split('/')[1] == split]
        items.sort(key=lambda kr: kr[1])
        if LIMIT:
            items = items[:LIMIT]
        keys = [k for k, _ in items]
        rows = np.array([r for _, r in items], dtype=np.int64)
        true_cls = [k.split('/')[0] for k in keys]
        n = len(rows)
        print(f'\n=== split={split}  n={n} ===')

        preds_word = []
        with torch.no_grad():
            for i in tqdm(range(0, n, BATCH), desc=f'decode {split}', unit='batch'):
                w = np.asarray(wav[rows[i:i + BATCH]], dtype=np.float32)
                audio = torch.from_numpy(w).to(device)
                audio = whisper.pad_or_trim(audio, whisper.audio.N_SAMPLES)  # -> 30 s
                mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)       # (B,80,3000)
                for r in whisper.decode(model, mel, options):
                    t = r.text.strip().split()
                    preds_word.append(norm(t[0]) if t else '')

        uniq = set(preds_word)
        nearest = {}
        for w in uniq:
            if not w:
                nearest[w] = None
            elif w in norm_to_class:
                nearest[w] = norm_to_class[w]
            else:
                mt = difflib.get_close_matches(w, classes_norm, n=1, cutoff=0.0)
                nearest[w] = norm_to_class[mt[0]] if mt else None

        exact = sum(1 for w, tc in zip(preds_word, true_cls) if w == norm(tc))
        near = sum(1 for w, tc in zip(preds_word, true_cls) if nearest[w] == tc)
        empty = sum(1 for w in preds_word if not w)
        print(f'  exact-match top1 : {exact/n:.4f}')
        print(f'  nearest-class top1: {near/n:.4f}')
        print(f'  empty/one-word rate: {empty/n:.4f}')
        print(f'  examples: ' + ', '.join(f'{tc}->{pw or "<empty>"}'
                                          for tc, pw in list(zip(true_cls, preds_word))[:8]))
        results_rows.append([split, n, f'{exact/n:.4f}', f'{near/n:.4f}', f'{empty/n:.4f}'])

    with open(os.path.join(HERE, 'results', 'asr_baseline.csv'), 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['split', 'n', 'exact_top1', 'nearest_top1', 'empty_rate'])
        wtr.writerows(results_rows)
    print('\nwrote asr_baseline.csv')


if __name__ == '__main__':
    main()
