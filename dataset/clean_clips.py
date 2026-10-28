"""Reject or repair mined clips using scan_clips.py metrics + the frames themselves (resumable).

    python dataset/clean_clips.py --dry      # decisions -> clean.csv only, nothing moved
    python dataset/clean_clips.py            # also: repair edge cuts, move rejects to clips/_rejected/,
                                             # rewrite clips.jsonl, append rejects to rejected.jsonl

Per clip (29 frames):
  cut     camera cut (grey frame difference >= CUT_D with colour histogram correlation < CUT_HIST, or difference
          >= CUT_D_HARD; motion and gestures keep the histogram) inside frames EDGE..28-EDGE -> reject;
          at the edges -> those frames are replaced by the nearest frame of the speaker's shot ("repaired")
  static  inner-lip opening range < MIN_OPEN_RNG (mouth hidden, e.g. head bowed over notes)
  profile median yaw > MAX_YAW (head turned fully sideways)
  offword re-aligned word centre more than MAX_OFF s from the clip centre (alignment error)
  outside mouth centre leaves the crop (head moves out of the fixed crop); only in edge frames -> repaired
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_clips import FPS, OUT, OUT_PX  # noqa: E402
from mine_bundestag import ffmpeg_exe  # noqa: E402
from word_stats import read_jsonl  # noqa: E402

CUT_D, CUT_HIST, CUT_D_HARD = 20.0, 0.85, 50.0   # mean abs grey difference (64x64), histogram correlation
EDGE = 3                         # edge frames that may be repaired
MIN_OPEN_RNG, MAX_YAW, MAX_OFF = 0.015, 0.9, 0.2
MOUTH_BOX = (0.1, 0.9)           # mouth centre must stay inside this part of the crop
CLEAN = OUT / "clean.csv"
REJ = OUT / "_rejected"


def frames_of(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)


def check(args):
    path, scan = args
    fr = frames_of(OUT / (path + ".mp4"))
    z = np.load(OUT / (path + ".npz"))
    lm = z["landmarks"].astype(np.float32)
    n = len(fr)
    if n != len(lm):
        return path, "reject", "broken", ""
    small = [cv2.resize(f, (64, 64), interpolation=cv2.INTER_AREA) for f in fr]
    g = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in small]).astype(np.float32)
    d = np.abs(np.diff(g, axis=0)).mean((1, 2))
    h = [cv2.normalize(cv2.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3), None).flatten() for f in small]
    hc = np.array([cv2.compareHist(h[i], h[i + 1], cv2.HISTCMP_CORREL) for i in range(n - 1)])
    cuts = [i + 1 for i in np.flatnonzero(((d >= CUT_D) & (hc < CUT_HIST)) | (d >= CUT_D_HARD))]  # new shot starts
    if any(EDGE <= c <= n - EDGE for c in cuts):
        return path, "reject", "cut", ""
    a = max([c for c in cuts if c < EDGE], default=0)        # main shot = frames a..b-1
    b = min([c for c in cuts if c > n - EDGE], default=n)
    ok = ~np.isnan(lm[:, 0, 0])
    mc = lm[:, [13, 14, 61, 291], :2].mean(1)
    out = ((mc < MOUTH_BOX[0]) | (mc > MOUTH_BOX[1])).any(1)                # NaN frames compare False
    while a < b and (not ok[a] or out[a]) and a < EDGE:      # edge frames without the face, or the mouth out of crop
        a += 1
    while b > a and (not ok[b - 1] or out[b - 1]) and b > n - EDGE:
        b -= 1
    main = lm[a:b][ok[a:b]]
    if scan is None:
        return path, "reject", "unscanned", ""
    if abs(float(scan["a_off"])) > MAX_OFF:
        return path, "reject", "offword", ""
    if float(scan["yaw_med"]) > MAX_YAW:
        return path, "reject", "profile", ""
    mw = np.hypot(main[:, 291, 0] - main[:, 61, 0], main[:, 291, 1] - main[:, 61, 1])
    op = np.hypot(main[:, 13, 0] - main[:, 14, 0], main[:, 13, 1] - main[:, 14, 1]) / mw
    if op.max() - op.min() < MIN_OPEN_RNG:
        return path, "reject", "static", ""
    if out[a:b].any():
        return path, "reject", "outside", ""
    fix = [i for i in range(n) if i < a or i >= b]
    return path, ("repair" if fix else "keep"), "", " ".join(map(str, fix))


def repair(path, fix):
    """Replace the given edge frames by the nearest kept frame (video + landmarks)."""
    fix = [int(i) for i in fix.split()]
    fr = frames_of(OUT / (path + ".mp4"))
    z = dict(np.load(OUT / (path + ".npz")))
    keep = [i for i in range(len(fr)) if i not in fix]
    for i in fix:
        j = min(keep, key=lambda k: abs(k - i))
        fr[i] = fr[j]
        z["landmarks"][i] = z["landmarks"][j]
    raw = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fr]).tobytes()
    tmp = OUT / (path + ".tmp.mp4")
    subprocess.run([ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{OUT_PX}x{OUT_PX}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "18", "-pix_fmt", "yuv420p", str(tmp)], input=raw, check=True)
    tmp.replace(OUT / (path + ".mp4"))
    np.savez_compressed(OUT / (path + ".npz"), **z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    scan = {r["path"]: r for r in csv.DictReader(open(OUT / "scan.csv", encoding="utf-8"))}
    clips = read_jsonl(OUT / "clips.jsonl")
    done = {r["path"] for r in csv.DictReader(open(CLEAN, encoding="utf-8"))} if CLEAN.exists() else set()
    todo = [c["path"] for c in clips if c["path"] not in done]
    print(f"{len(todo)} clips to check ({len(done)} done)", flush=True)
    with open(CLEAN, "a", encoding="utf-8", newline="") as f, ProcessPoolExecutor(args.workers) as ex:
        w = csv.writer(f)
        if not done:
            w.writerow(["path", "action", "reason", "fix_frames"])
        for i, row in enumerate(ex.map(check, [(p, scan.get(p)) for p in todo], chunksize=64)):
            w.writerow(row)
            if i % 20000 == 0:
                f.flush()
                print(f"{i}/{len(todo)}", flush=True)
    dec = {r["path"]: r for r in csv.DictReader(open(CLEAN, encoding="utf-8"))}
    acts = [dec[c["path"]]["action"] + " " + dec[c["path"]]["reason"] for c in clips]
    print({k: acts.count(k) for k in sorted(set(acts))}, flush=True)
    if args.dry:
        return

    todo_fix = [(c["path"], dec[c["path"]]["fix_frames"]) for c in clips if dec[c["path"]]["action"] == "repair"
                and not dec[c["path"]].get("repaired")]
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(repair, *zip(*todo_fix))) if todo_fix else None
    print(f"repaired {len(todo_fix)}", flush=True)

    kept, rej = [], []
    for c in clips:
        r = dec[c["path"]]
        if r["action"] == "reject":
            dst = REJ / c["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".mp4", ".npz"):
                shutil.move(str(OUT / (c["path"] + ext)), str(dst) + ext)
            rej.append({**c, "reject": r["reason"]})
        else:
            kept.append({**c, "repaired": r["fix_frames"]} if r["action"] == "repair" else c)
    shutil.copy(OUT / "clips.jsonl", OUT / "clips.jsonl.bak_clean")
    with open(OUT / "clips.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in kept)
    with open(OUT / "rejected.jsonl", "a", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in rej)
    with open(CLEAN, "w", encoding="utf-8", newline="") as f:  # mark repairs as applied
        w = csv.writer(f)
        w.writerow(["path", "action", "reason", "fix_frames"])
        for p, r in dec.items():
            w.writerow([p, "kept-repaired" if r["action"] == "repair" else r["action"], r["reason"], r["fix_frames"]])
    print(f"kept {len(kept)}, rejected {len(rej)} -> {REJ}", flush=True)


if __name__ == "__main__":
    main()
