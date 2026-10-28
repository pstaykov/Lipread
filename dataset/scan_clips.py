"""Quality metrics for every mined clip -> bundestag_data/clips/scan.csv (resumable, only unscanned clips).

    python dataset/scan_clips.py

Audio: the target word is aligned again on the clip's own 1.16 s of audio (MMS_FA with star tokens):
    a_score   mean frame probability of the word, a_off = word centre - clip centre (s).
Landmarks (crop-normalized, 29 frames):
    open_std / open_rng  inner-lip opening (13-14) / mouth width (61-291) over the clip; a covered mouth or a
                         silent face gives a static mouth
    yaw_med / yaw_max    nose tip (1) offset from the cheek midpoint (234, 454) / face width
    jump_max             largest frame-to-frame lip landmark move / face width (tracking jumps)
"""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_clips import CLIP, FPS, OUT, aligner, norm  # noqa: E402
from word_stats import read_jsonl  # noqa: E402

SCAN = OUT / "scan.csv"
COLS = ["path", "word", "id", "speaker_id", "a_score", "a_off", "n_nan", "open_std", "open_rng", "yaw_med",
        "yaw_max", "jump_max", "mouth_px"]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
BATCH = 64


def landmark_metrics(lm):
    lm = lm.astype(np.float32)
    ok = ~np.isnan(lm[:, 0, 0])
    p = lm[ok]
    d = lambda a, b: np.hypot(p[:, a, 0] - p[:, b, 0], p[:, a, 1] - p[:, b, 1])
    fw, mw = d(234, 454), d(61, 291)
    op = d(13, 14) / mw
    yaw = np.abs(p[:, 1, 0] - (p[:, 234, 0] + p[:, 454, 0]) / 2) / fw
    jumps = [0.0]
    for i in range(1, len(lm)):                            # consecutive valid frames only
        if ok[i] and ok[i - 1]:
            jumps.append(float(np.median(np.hypot(*(lm[i, LIPS, :2] - lm[i - 1, LIPS, :2]).T)) / fw.mean()))
    return dict(n_nan=int((~ok).sum()), open_std=float(op.std()), open_rng=float(op.max() - op.min()),
                yaw_med=float(np.median(yaw)), yaw_max=float(yaw.max()), jump_max=max(jumps))


def load(c):
    z = np.load(OUT / (c["path"] + ".npz"))
    return c, z["audio"], z["landmarks"]


def main():
    import torch
    done = {r["path"] for r in csv.DictReader(open(SCAN, encoding="utf-8"))} if SCAN.exists() else set()
    todo = [c for c in read_jsonl(OUT / "clips.jsonl") if c["path"] not in done]
    print(f"{len(todo)} clips to scan ({len(done)} done)", flush=True)
    model, tok, al = aligner()
    dev, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    n = int(CLIP / FPS * 16000)
    f = open(SCAN, "a", encoding="utf-8", newline="")
    w = csv.DictWriter(f, COLS)
    if not done:
        w.writeheader()
    with ThreadPoolExecutor(8) as ex:
        for b0 in range(0, len(todo), BATCH):
            items = list(ex.map(load, todo[b0:b0 + BATCH]))
            wav = np.zeros((len(items), n), np.float32)
            for i, (_, a, _) in enumerate(items):
                wav[i, :len(a)] = a[:n] / 32768
            with torch.inference_mode():                   # the torchaudio wrapper, per item (it only takes batch 1)
                x = torch.from_numpy(wav).to(dev)
                x = torch.nn.functional.layer_norm(x, x.shape[-1:]) if model.normalize_waveform else x
                em = torch.log_softmax(model.model(x.to(dtype))[0].float(), -1)
                em = torch.cat((em, torch.zeros_like(em[..., :1])), -1).cpu()
            r = n / em.shape[1] / 16000
            for i, (c, _, lm) in enumerate(items):
                try:
                    sp = al(em[i], tok(["*", norm(c["word"]), "*"]))[1]
                    sc = float(np.mean([x.score for x in sp]))
                    off = (sp[0].start + sp[-1].end) / 2 * r - n / 16000 / 2
                except Exception:
                    sc, off = 0.0, 9.9
                w.writerow({"path": c["path"], "word": c["word"], "id": c["id"], "speaker_id": c["speaker_id"],
                            "a_score": round(sc, 3), "a_off": round(off, 3), "mouth_px": c["mouth_px"],
                            **{k: round(v, 4) if isinstance(v, float) else v for k, v in landmark_metrics(lm).items()}})
            if b0 // BATCH % 200 == 0:
                f.flush()
                print(f"{b0 + len(items)}/{len(todo)}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
