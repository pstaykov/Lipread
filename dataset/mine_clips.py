"""Mine word clips for the fixed vocabulary from the Bundestag Mediathek HD files.

    python dataset/mine_clips.py --workers 6            # resumable; stops when every word has CAP clips
    taskkill /PID <bundestag_data/clips/miner.pid> /T /F # stop (kills the worker processes too)
    kill -- -$(cat bundestag_data/clips/miner.pid)       # stop on Linux (the miner leads its process group)
    python dataset/mine_clips.py --shard 1/2            # this machine mines only speakers with md5 % 2 == 1,
                                                        # up to CAP / 2 clips per word (see dataset/vm/)
    python dataset/mine_clips.py --ids DE-0210008043    # test on given speeches

Per speech (1-2 in main-process threads with one GPU model, 3-4 in CPU worker processes):
    1. Mediathek share data -> best MP4 (1920x1080 from Sept 2016) + MP3 audio.
    2. Sentences containing still-needed vocab words are force-aligned (torchaudio MMS_FA, GPU)
       on the MP3, padded by PAD s (OPTV sentence times are ~1 s early); words below MIN_SCORE drop.
    3. Only the 29-frame windows around those words are decoded from the HD file (seek, 50 fps
       interlaced files are deinterlaced with bwdif), cropped to the static centre ROI, FaceMesh-gated
       with the same lectern gate as mine_bundestag.py (geometry + per-speech wall-strip scene check).
    4. A clip is kept if >= 27/29 frames pass, the face does not drift and the mouth is >= MIN_MOUTH_PX
       wide in source pixels. Saved as a 256x256 face crop (GLips framing: crop = 2 x face width).
Output: bundestag_data/clips/<word>/<speech>_<frame>.{mp4,npz}, clips.jsonl, clips_index.jsonl.
"""
import argparse
import contextlib
import hashlib
import multiprocessing
import csv
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from collections import Counter, defaultdict
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from mine_bundestag import (DATA, FPS, GATE_CX, GATE_CY, GATE_W, MAX_GAP, MAX_RUN_JITTER, PIPE_H, PIPE_W,
                            ROI_H, ROI_W, ROI_X0, SCENE_TOL, STRIPS_X, STRIPS_Y, ffmpeg_exe, runs_of)
from word_stats import HD_FROM, read_jsonl

OUT = DATA / "clips"
SHARE = "https://webtv.bundestag.de/player/macros/_x_s-144277506/shareData.json?contentId={}"
UA = {"User-Agent": "Mozilla/5.0"}
CLIP = 29                     # frames, 1.16 s, word centred (as GLips / LRW)
PAD = 1.5                     # s of audio around each sentence for alignment
MIN_SCORE = 0.5               # mean CTC frame probability of the word
MIN_MOUTH_PX = 36             # lip-corner distance in source pixels
MIN_VALID = 27                # of 29 frames
CROP_FACE = 2.0               # crop side / face width (landmarks 234-454) -> mouth ~18 % of the crop
OUT_PX = 256
CAP = 800                     # clips per word (500 needed, headroom for later filtering)
SPK_CAP = 25                  # clips per word per speaker
SPEECH_CAP = 5                # clips per word per speech
MAX_SEG = 250                 # frames per decode segment (10 s of 768x540 RGB ~ 300 MB)
MIN_FREE_GB = 30.0
PREP_THREADS = 3              # download + alignment threads in the main process (GPU shared via a lock)
CPU_ALIGN_THREADS = 2         # torch threads per alignment without GPU (prep threads align in parallel)
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")

_model = None


def norm(w):
    w = w.lower().replace("ä", "a").replace("ö", "o").replace("ü", "u").replace("ß", "ss")
    return re.sub(r"[^a-z]", "", w)


def media_urls(origin_id):
    with urllib.request.urlopen(urllib.request.Request(SHARE.format(origin_id), headers=UA), timeout=60) as r:
        t = r.read().decode("utf-8", "replace")
    mp4 = re.findall(r'"(https://[^"]+_h264_(\d+)_(\d+)_(\d+)kb[^"]*\.mp4[^"]*)"', t)
    mp3 = re.findall(r'"audioUrlStereo"\s*:\s*"(https://[^"]+\.mp3[^"]*)"', t)
    if not mp4 or not mp3:
        return None, None
    return max(mp4, key=lambda x: (int(x[1]), int(x[3])))[0], mp3[0]


def probe(url):
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", url], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    m = re.search(r"Video:.*?(\d{3,4})x(\d{3,4}).*?([\d.]+) fps", r.stderr)
    return (int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else None


def pcm16k(url):
    r = subprocess.run([ffmpeg_exe(), "-v", "error", "-nostdin", "-i", url, "-f", "s16le", "-ac", "1",
                        "-ar", "16000", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.int16)


def shard_of(speaker_id, n):
    """Stable across machines (unlike hash())."""
    return int(hashlib.md5(speaker_id.encode()).hexdigest(), 16) % n


def init_worker():
    os.environ.setdefault("GLOG_minloglevel", "2")              # workers never import torch
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def aligner():
    global _model
    if _model is None:
        import torch
        import torchaudio
        b = torchaudio.pipelines.MMS_FA
        m = b.get_model(with_star=True).eval()
        if torch.cuda.is_available():
            m = m.cuda().half()                            # fp16: same word times, ~3x faster
        else:
            torch.set_num_threads(CPU_ALIGN_THREADS)
        _model = (m, b.get_tokenizer(), b.get_aligner())
    return _model


def align_words(pcm, entry, needed):
    """Word times for needed vocab words: [(word, start, end, score)]."""
    import torch
    model, tok, al = aligner()
    p = next(model.parameters())
    dev, dtype = p.device, p.dtype
    wav = torch.from_numpy(pcm.astype(np.float32) / 32768)
    out = []
    for s, e, text in entry["sentences"]:
        words = [(w.lower(), norm(w)) for w in WORD.findall(text)]
        words = [(w, n) for w, n in words if n]
        if not any(w in needed for w, _ in words):
            continue
        a = max(0.0, s - PAD)
        seg = wav[int(a * 16000):int((e + PAD) * 16000)]
        if not 16000 <= len(seg) <= 45 * 16000:              # very long sentences: GPU memory
            continue
        with torch.inference_mode():
            em, _ = model(seg[None].to(dev, dtype))
            em = em.float()
        try:
            spans = al(em[0], tok(["*"] + [n for _, n in words] + ["*"]))[1:-1]
        except Exception:                                  # transcript longer than audio etc.
            continue
        r = len(seg) / em.shape[1] / 16000
        for (w, _), sp in zip(words, spans):
            if w in needed:
                score = float(np.mean([x.score for x in sp]))
                t0, t1 = a + sp[0].start * r, a + sp[-1].end * r
                if score >= MIN_SCORE and 0.08 <= t1 - t0 <= 1.2:
                    out.append((w, t0, t1, score))
    return out


def decode(url, f0, n, interlaced, roi_wh):
    """n native-resolution ROI frames starting at frame f0 (25 fps timeline)."""
    W, H = roi_wh
    vf = ("bwdif=mode=send_frame," if interlaced else "") + \
         f"fps={FPS},crop={W}:{H}:trunc(iw*{ROI_X0 / 2})*2:0"
    cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", "-reconnect", "1", "-reconnect_delay_max", "10",
           "-ss", f"{f0 / FPS:.3f}", "-i", url, "-frames:v", str(n), "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    k = len(raw) // (W * H * 3)
    return np.frombuffer(raw[:k * W * H * 3], np.uint8).reshape(k, H, W, 3)


def analyse(frames, fm):
    """Per frame: landmarks of the single face passing the lectern geometry (or None), strip vector."""
    import cv2
    lms, strips = [], []
    y0s, y1s = int(STRIPS_Y[0] * PIPE_H), int(STRIPS_Y[1] * PIPE_H)
    for fr in frames:
        small = cv2.resize(fr, (PIPE_W, PIPE_H), interpolation=cv2.INTER_AREA)
        strips.append(np.concatenate([small[y0s:y1s, int(a * PIPE_W):int(b * PIPE_W)][::4, ::4].reshape(-1, 3)
                                      for a, b in STRIPS_X]).astype(np.float32))
        passing = []
        for lm in fm.process(small).multi_face_landmarks or []:
            p = np.array([[l.x, l.y, l.z] for l in lm.landmark], np.float32)
            (x0, y0), (x1, y1) = p[:, :2].min(0), p[:, :2].max(0)
            cx, cy, w = ROI_X0 + ROI_W * (x0 + x1) / 2, ROI_H * (y0 + y1) / 2, ROI_W * (x1 - x0)
            if GATE_CX[0] <= cx <= GATE_CX[1] and GATE_CY[0] <= cy <= GATE_CY[1] and GATE_W[0] <= w <= GATE_W[1]:
                passing.append(p)
        lms.append(passing[0] if len(passing) == 1 else None)
    return lms, strips


def write_clip(path, frames, lms, roi_wh, info, pcm):
    """Fixed per-clip face crop (side = CROP_FACE x median face width) -> 256x256 mp4 + npz."""
    import cv2
    W, H = roi_wh
    ok = [p for p in lms if p is not None]
    fw = np.median([np.hypot((p[454, 0] - p[234, 0]) * W, (p[454, 1] - p[234, 1]) * H) for p in ok])
    cx = np.median([(p[:, 0].min() + p[:, 0].max()) / 2 * W for p in ok])
    cy = np.median([(p[:, 1].min() + p[:, 1].max()) / 2 * H for p in ok])
    side = int(round(CROP_FACE * fw))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    pad = side
    crops = []
    for fr in frames:
        big = cv2.copyMakeBorder(fr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        crops.append(cv2.resize(big[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side], (OUT_PX, OUT_PX),
                                interpolation=cv2.INTER_AREA))
    cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{OUT_PX}x{OUT_PX}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "18", "-pix_fmt", "yuv420p", str(path.with_suffix(".mp4"))]
    subprocess.run(cmd, input=np.stack(crops).tobytes(), check=True)
    lm = np.full((len(lms), 478, 3), np.nan, np.float32)
    for i, p in enumerate(lms):
        if p is not None:                                  # ROI-normalized -> crop-normalized
            lm[i, :, 0] = (p[:, 0] * W - x0) / side
            lm[i, :, 1] = (p[:, 1] * H - y0) / side
            lm[i, :, 2] = p[:, 2] * W / side
    a = int(info["f0"] / FPS * 16000)
    np.savez_compressed(path.with_suffix(".npz"), landmarks=lm.astype(np.float16),
                        audio=pcm[a:a + int(CLIP / FPS * 16000)],
                        crop_box=np.array([x0, y0, side], np.int32), roi_px=np.array(roi_wh, np.int32))


def prepare_speech(entry, needed, lock):
    """Main process (threads): media URLs, probe, MP3, alignment on the single GPU model -> worker job."""
    sid, t_start = entry["id"], time.time()
    base = {"id": sid, "speaker_id": entry["speaker_id"]}
    try:
        mp4, mp3 = media_urls(entry["origin_media_id"])
        if not mp4:
            return {**base, "status": "error", "error": "no media urls"}, None
        pr = probe(mp4)
        if not pr or pr[1] < 720:
            return {**base, "status": "skipped", "error": f"resolution {pr}"}, None
        vw, vh, fps = pr
        pcm = pcm16k(mp3)
        with lock:
            words = align_words(pcm, entry, set(needed))
    except Exception as ex:
        return {**base, "status": "error", "error": f"{type(ex).__name__}: {ex}"[:300]}, None

    by_word = defaultdict(list)                            # best SPEECH_CAP instances per word
    for w, t0, t1, sc in words:
        by_word[w].append((sc, t0, t1))
    cands = []
    for w, lst in by_word.items():
        taken = []                                         # OPTV sometimes repeats a sentence -> same instance twice
        for sc, t0, t1 in sorted(lst, reverse=True):
            f0 = int(round((t0 + t1) / 2 * FPS)) - CLIP // 2
            if f0 < 0 or any(abs(f0 - g) < 5 for g in taken):
                continue
            taken.append(f0)
            cands.append({"word": w, "f0": f0, "t0": round(t0, 3), "t1": round(t1, 3), "score": round(sc, 3)})
            if len(taken) == SPEECH_CAP:
                break
    base.update(aligned=len(words), candidates=len(cands), src=f"{vw}x{vh}", prep_sec=round(time.time() - t_start, 1))
    if not cands:
        return {**base, "status": "done", "clips": 0}, None
    return base, {"entry": entry, "mp4": mp4, "interlaced": fps > 30,      # 1080 at 50 fps = interlaced fields
                  "roi_wh": (int(vw * ROI_W) // 2 * 2, int(vh * ROI_H) // 2 * 2), "pcm": pcm, "cands": cands,
                  "src": f"{vw}x{vh}"}


def mine_speech(base, job):
    """Worker (CPU only, no torch): decode the clip windows, FaceMesh-gate, write clips."""
    import mediapipe as mp
    t_start = time.time()
    entry, mp4, interlaced, roi_wh, pcm = job["entry"], job["mp4"], job["interlaced"], job["roi_wh"], job["pcm"]
    sid = entry["id"]

    # merge clip windows into decode segments of at most MAX_SEG frames (bounded memory)
    segs = []
    for c in sorted(job["cands"], key=lambda c: c["f0"]):
        if segs and c["f0"] <= segs[-1][1] + FPS and c["f0"] + CLIP - segs[-1][0] <= MAX_SEG:
            segs[-1][1] = max(segs[-1][1], c["f0"] + CLIP)
            segs[-1][2].append(c)
        else:
            segs.append([c["f0"], c["f0"] + CLIP, [c]])

    tmp = OUT / "_tmp" / sid
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    pending, all_strips, all_ok = [], [], []
    for s, e, cs in segs:
        frames = decode(mp4, s, e - s, interlaced, roi_wh)
        if len(frames) < e - s:
            continue
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2, refine_landmarks=True,
                                             min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
            lms, strips = analyse(frames, fm)
        all_strips += strips
        all_ok += [p is not None for p in lms]
        for c in cs:
            i = c["f0"] - s
            c_lms, c_strips = lms[i:i + CLIP], strips[i:i + CLIP]
            if sum(p is not None for p in c_lms) < MIN_VALID:
                continue
            path = tmp / f"{c['word']}_{c['f0']}"
            write_clip(path, frames[i:i + CLIP], c_lms, roi_wh, c, pcm)
            pending.append((c, path, c_lms, c_strips))
        del frames

    # scene check against the speech's reference strip, then per-clip checks
    clips = []
    if sum(all_ok) >= 2 * CLIP:
        ref = np.median(np.stack([st for st, ok in zip(all_strips, all_ok) if ok]), 0)
        W, H = roi_wh
        for c, path, c_lms, c_strips in pending:
            good = np.array([p is not None and np.abs(st - ref).mean() < SCENE_TOL for p, st in zip(c_lms, c_strips)])
            gaps = [e - s for s, e in runs_of(~good)]
            ok = [p for p, g in zip(c_lms, good) if g]
            if good.sum() < MIN_VALID or (gaps and max(gaps) > MAX_GAP):
                continue
            centres = np.array([[(p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2] for p in ok])
            mouth = float(np.median([np.hypot((p[291, 0] - p[61, 0]) * W, (p[291, 1] - p[61, 1]) * H) for p in ok]))
            if (centres * (ROI_W, ROI_H)).std(0).max() > MAX_RUN_JITTER or mouth < MIN_MOUTH_PX:
                continue
            dst = OUT / c["word"] / f"{sid}_{c['f0']}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            for ext in (".mp4", ".npz"):
                shutil.move(str(path.with_suffix(ext)), str(dst.with_suffix(ext)))
            clips.append({**c, "id": sid, "speaker_id": entry["speaker_id"],
                          "path": str(dst.relative_to(OUT)).replace("\\", "/"),
                          "mouth_px": round(mouth, 1), "src": job["src"], "deinterlaced": interlaced,
                          "date": entry["date"], "period": entry["period"]})
    shutil.rmtree(tmp, ignore_errors=True)
    return {**base, "status": "done", "clips": len(clips), "sec": round(time.time() - t_start, 1)}, clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--vocab", default=str(DATA / "words" / "vocab.csv"))
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--shard", default="0/1", help="k/n: only speakers with md5 %% n == k, CAP / n clips per word")
    ap.add_argument("--prep", type=int, help="download + alignment threads (default: 3 with GPU, workers/4 without)")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))
    cap = CAP // n
    import torch                                            # main process only; decides GPU vs CPU alignment
    gpu = torch.cuda.is_available()
    prep_threads = args.prep or (PREP_THREADS if gpu else max(PREP_THREADS, args.workers // 4))
    OUT.mkdir(parents=True, exist_ok=True)

    vocab = {r["word"].lower(): int(r["est"]) for r in csv.DictReader(open(args.vocab, encoding="utf-8"))}
    done = {r["id"] for r in read_jsonl(OUT / "clips_index.jsonl")}
    count, spk = Counter(), Counter()
    registered = set()
    for c in read_jsonl(OUT / "clips.jsonl"):
        count[c["word"]] += 1
        spk[(c["word"], c["speaker_id"])] += 1
        registered.add(c["path"])
    # files of clips that a stopped run moved but never recorded
    shutil.rmtree(OUT / "_tmp", ignore_errors=True)
    stale = [f for w in vocab if (OUT / w).is_dir() for f in (OUT / w).iterdir() if f"{w}/{f.stem}" not in registered]
    for f in stale:
        f.unlink()
    if stale:
        print(f"removed {len(stale)} unregistered clip files", flush=True)

    entries = [e for e in read_jsonl(DATA / "catalog.jsonl") if e["date"] >= HD_FROM and e["id"] not in done
               and shard_of(e["speaker_id"], n) == k]
    if args.ids:
        entries = [e for e in entries if e["id"] in set(args.ids)]
    occ = {}
    for e in entries:
        occ[e["id"]] = Counter(w.lower() for _, _, t in e["sentences"] for w in WORD.findall(t) if w.lower() in vocab)
    # rare words first: occurrences weighted by 1 / estimated clips, per minute
    entries.sort(key=lambda e: -sum(n / vocab[w] for w, n in occ[e["id"]].items()) / max(e["duration"], 1))
    print(f"{len(entries)} speeches to go (shard {k}/{n}, cap {cap}), {len(vocab)} words, "
          f"{sum(count.values())} clips so far, alignment on {'GPU' if gpu else 'CPU'}, {prep_threads} prep threads",
          flush=True)

    if os.name == "posix":
        with contextlib.suppress(OSError):
            os.setpgrp()                                    # workers join our group -> kill -- -<pid> stops all
    (OUT / "miner.pid").write_text(str(os.getpid()))        # stop: taskkill /PID <pid> /T /F (whole tree)
    idx = open(OUT / "clips_index.jsonl", "a", encoding="utf-8")
    cl = open(OUT / "clips.jsonl", "a", encoding="utf-8")

    def record(e, res, clips):
        kept = 0
        for c in clips:
            key = (c["word"], c["speaker_id"])
            if count[c["word"]] >= cap or spk[key] >= SPK_CAP:
                for ext in (".mp4", ".npz"):
                    (OUT / c["path"]).with_suffix(ext).unlink(missing_ok=True)
                continue
            count[c["word"]] += 1
            spk[key] += 1
            kept += 1
            cl.write(json.dumps(c, ensure_ascii=False) + "\n")
        res["clips"] = kept
        idx.write(json.dumps(res, ensure_ascii=False) + "\n")
        cl.flush()
        idx.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {res['status']:7s} {e['id']} {e['speaker'][:24]:24s} "
              f"clips={kept:3d} prep={res.get('prep_sec', '')}s mine={res.get('sec', '')}s | "
              f"{sum(count.values())} clips, {sum(count[w] >= 500 for w in vocab)}/{len(vocab)} words >= 500, "
              f"min {min(count[w] for w in vocab)}", flush=True)

    lock = threading.Lock() if gpu else contextlib.nullcontext()   # one GPU model, one thread at a time
    it = iter(entries)
    spawn = multiprocessing.get_context("spawn")            # no fork of a process with threads + torch (Linux)
    with ThreadPoolExecutor(prep_threads) as prep,             ProcessPoolExecutor(args.workers, mp_context=spawn, initializer=init_worker) as ex:
        preparing, mining = {}, {}
        while True:
            full = all(count[w] >= cap for w in vocab)
            low_disk = shutil.disk_usage(OUT).free / 1e9 < MIN_FREE_GB
            while not full and not low_disk and len(preparing) < prep_threads and len(mining) < 2 * args.workers:
                e = next(it, None)
                if e is None:
                    break
                needed = [w for w in occ[e["id"]] if count[w] < cap and spk[(w, e["speaker_id"])] < SPK_CAP]
                if needed:
                    preparing[prep.submit(prepare_speech, e, needed, lock)] = e
            if not preparing and not mining:
                break
            fin, _ = wait(list(preparing) + list(mining), return_when=FIRST_COMPLETED)
            for f in fin:
                if f in preparing:
                    e = preparing.pop(f)
                    try:
                        base, job = f.result()
                    except Exception as exc:
                        base, job = {"id": e["id"], "status": "error", "error": repr(exc)[:300]}, None
                    if job is None:
                        record(e, base, [])
                    else:
                        mining[ex.submit(mine_speech, base, job)] = e
                else:
                    e = mining.pop(f)
                    try:
                        res, clips = f.result()
                    except BrokenProcessPool:                # worker killed (e.g. stopped): retry next run
                        print(f"interrupted {e['id']}", flush=True)
                        continue
                    except Exception as exc:
                        res, clips = {"id": e["id"], "status": "error", "error": repr(exc)[:300]}, []
                    record(e, res, clips)
    print("stopped:", "all words full" if all(count[w] >= cap for w in vocab) else "no more speeches / disk", flush=True)


if __name__ == "__main__":
    main()
