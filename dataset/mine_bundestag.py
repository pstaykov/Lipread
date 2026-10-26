"""Mine Bundestag speeches (via the Open Parliament TV API) into a lip-reading corpus.

Two steps:
    python dataset/mine_bundestag.py catalog        # list candidate speeches (API only, no video)
    python dataset/mine_bundestag.py mine           # stream, crop, FaceMesh-gate, store

Both steps resume where they stopped. See dataset/MINING.md for the approach.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bundestag_data"
API = "https://de.openparliament.tv/api/v1"
UA = {"User-Agent": "BWKI-JuFo-lipreading-research (peter.sta.2010@gmail.com)"}

# Fixed centre ROI in full-frame normalized coords (camera is static on the lectern shot).
ROI_X0, ROI_W, ROI_H = 0.30, 0.40, 0.50
PIPE_W, PIPE_H = 512, 360           # analysis copy of the ROI (16:9 source -> 0.4*16 : 0.5*9)
FPS = 25

# Lectern-shot gate, measured on P19/P21 samples (face bbox, full-frame normalized).
# Lectern: cx 0.44-0.55, cy 0.16-0.33, w 0.08-0.12. President/audience close-ups: w >= 0.13, cy >= 0.4.
GATE_CX = (0.38, 0.62)
GATE_CY = (0.08, 0.40)
GATE_W = (0.06, 0.125)
# Scene check: wall strips left/right of the head (ROI-normalized). On the static lectern shot they
# look the same in every frame, so frames far from the speech's median strip are a different shot.
STRIPS_X = ((0.0, 0.12), (0.88, 1.0))
STRIPS_Y = (0.30, 0.50)
SCENE_TOL = 12.0                    # mean abs RGB difference to the per-speech reference
MIN_RUN = 13                       # frames (~0.5 s) a valid run must last
MAX_GAP = 2                         # frames of detector dropout bridged inside a run
MAX_RUN_JITTER = 0.05               # std of face centre within a run; above = camera moved

# A speech is kept only if it gives enough usable face-on-screen speech.
MIN_VALID_SEC = 20.0
MIN_VALID_FRAC = 0.25

BUDGET_GB = 250.0
MIN_FREE_GB = 60.0


def ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def api_get(path, **params):
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  api retry {attempt + 1}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


# ---------------------------------------------------------------- catalog

def speech_entry(x):
    """Return a catalog entry for a media item, or None if it is not a usable single-speaker speech."""
    a = x["attributes"]
    if not a.get("aligned") or not a.get("videoFileURI") or not a.get("public", True):
        return None
    if not (60 <= (a.get("duration") or 0) <= 1800):
        return None
    main = {p["id"] for p in x["annotations"]["data"]
            if p["type"] == "person" and p["attributes"].get("context") == "main-speaker"}
    if len(main) != 1:
        return None
    sid = main.pop()
    person = next((p for p in x["relationships"]["people"]["data"] if p["id"] == sid), None)
    if person is None:
        return None
    pa = person["attributes"]
    sentences = []
    for tc in a.get("textContents", []):
        if tc.get("type") != "proceedings":
            continue
        for tb in tc["textBody"]:
            if tb["type"] != "speech" or tb["speakerstatus"] != "main-speaker":
                continue
            for s in tb.get("sentences", []):
                try:
                    sentences.append([float(s["timeStart"]), float(s["timeEnd"]), s["text"]])
                except (TypeError, ValueError, KeyError):
                    pass
    speech_sec = sum(max(0.0, e - s) for s, e, _ in sentences)
    if speech_sec < 30:
        return None
    rel = x["relationships"]
    return {
        "id": x["id"],
        "origin_media_id": a.get("originMediaID"),
        "video": a["videoFileURI"],
        "source_page": a.get("sourcePage"),
        "date": a["dateStart"],
        "duration": a["duration"],
        "period": rel["electoralPeriod"]["data"]["attributes"]["number"],
        "session": rel["session"]["data"]["id"],
        "agenda": (rel.get("agendaItem", {}).get("data") or {}).get("attributes", {}).get("title"),
        "speaker_id": sid,
        "speaker": pa.get("label"),
        "speaker_type": pa.get("type"),
        "party": (pa.get("party") or {}).get("label"),
        "faction": (pa.get("faction") or {}).get("label"),
        "speech_sec": round(speech_sec, 2),
        "sentences": sentences,
    }


def cmd_catalog(args):
    DATA.mkdir(exist_ok=True)
    cat_path = DATA / "catalog.jsonl"
    done_path = DATA / "catalog_sessions.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    with open(cat_path, "a", encoding="utf-8") as cat, open(done_path, "a") as dn:
        for period in args.periods:
            empty_run = 0
            n = 0
            while empty_run < 5:
                n += 1
                sess = f"DE-0{period:02d}{n:04d}"
                if sess in done:
                    empty_run = 0
                    continue
                items, page = [], 1
                while True:
                    d = api_get("search/media", parliament="DE", sessionID=sess, limit=100, page=page)
                    items += d.get("data", [])
                    if len(items) >= d["meta"]["results"]["total"] or not d.get("data"):
                        break
                    page += 1
                if not items:
                    empty_run += 1
                    continue
                empty_run = 0
                kept = 0
                for x in items:
                    try:
                        e = speech_entry(x)
                    except (KeyError, TypeError, AttributeError):   # incomplete API record
                        e = None
                    if e:
                        cat.write(json.dumps(e, ensure_ascii=False) + "\n")
                        kept += 1
                cat.flush()
                dn.write(sess + "\n")
                dn.flush()
                print(f"{sess}: {len(items)} media, {kept} speeches", flush=True)
                time.sleep(args.sleep)


# ---------------------------------------------------------------- mining

def runs_of(mask):
    """(start, end) half-open index pairs of True runs."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def clean_mask(det, centers):
    """Bridge short dropouts, drop short runs and runs where the face drifts (camera move)."""
    mask = det.copy()
    for s, e in runs_of(~det):
        if s > 0 and e < len(det) and e - s <= MAX_GAP:
            mask[s:e] = True
    out = np.zeros_like(mask)
    for s, e in runs_of(mask):
        if e - s < MIN_RUN:
            continue
        c = centers[s:e][det[s:e]]
        if len(c) and c.std(0).max() > MAX_RUN_JITTER:
            continue
        out[s:e] = True
    return out


def process_speech(entry, out_root, threads):
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import mediapipe as mp

    sid = entry["id"]
    final = out_root / sid
    tmp = out_root / (sid + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t0 = time.time()

    crop = (f"crop=trunc(iw*{ROI_W / 2})*2:trunc(ih*{ROI_H / 2})*2:"
            f"trunc(iw*{ROI_X0 / 2})*2:0")
    net = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10"] \
        if entry["video"].startswith("http") else []
    cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", *net,
           "-i", entry["video"],
           "-filter_complex", f"[0:v]fps={FPS},{crop},split=2[a][b];[b]scale={PIPE_W}:{PIPE_H}[p]",
           "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
           "-pix_fmt", "yuv420p", "-threads", str(threads), str(tmp / "roi.mp4"),
           "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(tmp / "audio.flac"),
           "-map", "[p]", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    err_path = tmp / "ffmpeg.err"
    with open(err_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, bufsize=PIPE_W * PIPE_H * 3 * 4)

        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2,
                                             refine_landmarks=True,
                                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
        fsize = PIPE_W * PIPE_H * 3
        boxes, lms, lm_idx, scene = [], [], [], []
        y0s, y1s = int(STRIPS_Y[0] * PIPE_H), int(STRIPS_Y[1] * PIPE_H)
        i = 0
        while True:
            buf = proc.stdout.read(fsize)
            if len(buf) < fsize:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(PIPE_H, PIPE_W, 3)
            scene.append(np.concatenate([
                frame[y0s:y1s, int(a * PIPE_W):int(b * PIPE_W)][::4, ::4].reshape(-1, 3)
                for a, b in STRIPS_X]))
            res = fm.process(frame)
            passing = []
            for lm in res.multi_face_landmarks or []:
                p = np.array([[l.x, l.y, l.z] for l in lm.landmark], np.float32)
                x0, y0 = p[:, :2].min(0)
                x1, y1 = p[:, :2].max(0)
                # ROI-normalized -> full-frame normalized
                cx = ROI_X0 + ROI_W * (x0 + x1) / 2
                cy = ROI_H * (y0 + y1) / 2
                w = ROI_W * (x1 - x0)
                ok = (GATE_CX[0] <= cx <= GATE_CX[1] and GATE_CY[0] <= cy <= GATE_CY[1]
                      and GATE_W[0] <= w <= GATE_W[1])
                if ok:
                    passing.append((p, (cx, cy, w, ROI_H * (y1 - y0))))
            hit = passing[0] if len(passing) == 1 else None   # two faces in the gate -> ambiguous
            if hit is not None:
                lms.append(hit[0].astype(np.float16))
                lm_idx.append(i)
                boxes.append(hit[1])
            else:
                boxes.append((np.nan,) * 4)
            i += 1
        fm.close()
        proc.stdout.close()
        rc = proc.wait()

    n = i
    if rc != 0 or n == 0:
        msg = err_path.read_text(errors="replace")[-500:]
        shutil.rmtree(tmp, ignore_errors=True)
        return {"id": sid, "status": "error", "error": f"ffmpeg rc={rc} frames={n}: {msg}"}

    boxes = np.array(boxes, np.float32)
    det = ~np.isnan(boxes[:, 0])
    scene = np.stack(scene).astype(np.float32)
    ref = np.median(scene[det], 0) if det.any() else scene.mean(0)
    scene_dist = np.abs(scene - ref).mean((1, 2))
    det &= scene_dist < SCENE_TOL
    valid = clean_mask(det, np.nan_to_num(boxes[:, :2]))

    speech = np.zeros(n, bool)
    for s, e, _ in entry["sentences"]:
        speech[int(s * FPS):int(np.ceil(e * FPS))] = True
    speech = speech[:n]
    usable = valid & speech
    valid_sec = usable.sum() / FPS
    frac = usable.sum() / max(1, speech.sum())

    stats = {"frames": int(n), "detected": int(det.sum()), "valid": int(valid.sum()),
             "usable_sec": round(float(valid_sec), 2), "usable_frac": round(float(frac), 3),
             "proc_sec": round(time.time() - t0, 1)}
    if valid_sec < MIN_VALID_SEC or frac < MIN_VALID_FRAC:
        shutil.rmtree(tmp, ignore_errors=True)
        return {"id": sid, "speaker_id": entry["speaker_id"], "status": "rejected", **stats}

    import cv2
    cap = cv2.VideoCapture(str(tmp / "roi.mp4"))
    roi_px = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    np.savez_compressed(tmp / "facemesh.npz", roi_px=np.array(roi_px, np.int32),
                        landmarks=np.stack(lms) if lms else np.zeros((0, 478, 3), np.float16),
                        landmark_frame=np.array(lm_idx, np.int32),
                        bbox=boxes.astype(np.float16), scene_dist=scene_dist.astype(np.float16),
                        valid=valid, speech=speech)
    meta = {k: v for k, v in entry.items()}
    meta.update({"fps": FPS, "roi": {"x0": ROI_X0, "y0": 0.0, "w": ROI_W, "h": ROI_H},
                 "roi_px": {"w": roi_px[0], "h": roi_px[1]},
                 "gate": {"cx": GATE_CX, "cy": GATE_CY, "w": GATE_W, "scene_tol": SCENE_TOL}, **stats})
    (tmp / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    err_path.unlink(missing_ok=True)
    shutil.rmtree(final, ignore_errors=True)
    for attempt in range(30):                   # Windows: AV/indexer may briefly lock new files
        try:
            tmp.rename(final)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(2)
    size = sum(f.stat().st_size for f in final.iterdir())
    return {"id": sid, "speaker_id": entry["speaker_id"], "status": "kept", "bytes": size, **stats}


def round_robin(entries, seed):
    """Order speeches so every speaker gets their 1st speech before anyone gets a 2nd, etc."""
    by = defaultdict(list)
    for e in entries:
        by[e["speaker_id"]].append(e)
    rng = random.Random(seed)
    speakers = sorted(by)
    rng.shuffle(speakers)
    for s in speakers:
        rng.shuffle(by[s])
    k = 0
    while True:
        layer = [by[s][k] for s in speakers if k < len(by[s])]
        if not layer:
            return
        yield from layer
        k += 1


def by_target_density(entries, targets_csv):
    """Order speeches by target-word occurrences per minute, rarer targets weighted higher."""
    import csv
    import re
    targets = {r["word"].lower() for r in csv.DictReader(open(targets_csv, encoding="utf-8"))}
    tok = re.compile(r"[A-Za-zÄÖÜäöüß]+")
    counts = {}
    total = defaultdict(int)
    for e in entries:
        c = defaultdict(int)
        for _, _, text in e["sentences"]:
            for w in tok.findall(text):
                w = w.lower()
                if w in targets:
                    c[w] += 1
                    total[w] += 1
        counts[e["id"]] = c
    score = {e["id"]: sum(n / np.sqrt(total[w]) for w, n in counts[e["id"]].items()) / max(e["duration"], 1)
             for e in entries}
    return sorted(entries, key=lambda e: -score[e["id"]])


def cmd_mine(args):
    out_root = DATA / "speeches"
    out_root.mkdir(parents=True, exist_ok=True)
    idx_path = DATA / "index.jsonl"
    entries = [json.loads(l) for l in open(DATA / "catalog.jsonl", encoding="utf-8")]
    entries = [e for e in entries if e["period"] in args.periods]
    entries = list({e["id"]: e for e in entries}.values())

    finished, kept_per, used = set(), defaultdict(int), 0
    if idx_path.exists():
        for l in open(idx_path, encoding="utf-8"):
            r = json.loads(l)
            if r["status"] in ("kept", "rejected"):
                finished.add(r["id"])
            if r["status"] == "kept":
                kept_per[r["speaker_id"]] += 1
                used += r["bytes"]
    for p in out_root.glob("*.part"):
        shutil.rmtree(p, ignore_errors=True)

    order = by_target_density(entries, args.targets) if args.targets else round_robin(entries, args.seed)
    queue = (e for e in order if e["id"] not in finished)
    budget = args.budget_gb * 1e9
    print(f"{len(entries)} catalog speeches, {len(finished)} done, {used / 1e9:.1f} GB used", flush=True)

    def can_start():
        free = shutil.disk_usage(DATA).free
        return used < budget and free > args.min_free_gb * 1e9

    with ProcessPoolExecutor(args.workers) as ex, open(idx_path, "a", encoding="utf-8") as idx:
        running = {}
        exhausted = False
        while True:
            while not exhausted and len(running) < args.workers and can_start():
                e = next(queue, None)
                if e is None:
                    exhausted = True
                    break
                if kept_per[e["speaker_id"]] >= args.max_per_speaker:
                    continue
                kept_per[e["speaker_id"]] += 1          # reserve; released if not kept
                running[ex.submit(process_speech, e, out_root, args.ffmpeg_threads)] = e
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                e = running.pop(f)
                try:
                    r = f.result()
                except Exception as exc:
                    r = {"id": e["id"], "status": "error", "error": repr(exc)}
                r.setdefault("speaker_id", e["speaker_id"])
                if r["status"] == "kept":
                    used += r["bytes"]
                else:
                    kept_per[e["speaker_id"]] -= 1
                idx.write(json.dumps(r, ensure_ascii=False) + "\n")
                idx.flush()
                n_spk = sum(1 for v in kept_per.values() if v > 0)
                print(f"[{time.strftime('%H:%M:%S')}] {r['status']:8s} {e['id']} {e['speaker']:<28.28s} "
                      f"usable={r.get('usable_sec', 0):6.1f}s t={r.get('proc_sec', 0):5.1f}s | "
                      f"{used / 1e9:6.2f} GB, {n_spk} speakers"
                      + (f" | {r['error'][:200]}" if r["status"] == "error" else ""), flush=True)
    reason = "budget reached" if used >= budget else "catalog exhausted / disk floor"
    print(f"stopped: {reason}, {used / 1e9:.1f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("catalog")
    c.add_argument("--periods", type=int, nargs="+", default=[21, 20, 19])
    c.add_argument("--sleep", type=float, default=0.5)
    m = sub.add_parser("mine")
    m.add_argument("--periods", type=int, nargs="+", default=[21, 20, 19])
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--ffmpeg-threads", type=int, default=2)
    m.add_argument("--max-per-speaker", type=int, default=10)
    m.add_argument("--budget-gb", type=float, default=BUDGET_GB)
    m.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--targets", help="targets.csv from select_targets.py; mine target-dense speeches first")
    args = ap.parse_args()
    {"catalog": cmd_catalog, "mine": cmd_mine}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
