"""Offline mouth-ROI preprocessing: crops each GLips clip to a tight, mouth-centered square (MediaPipe FaceMesh, EMA-smoothed),
resizes to --out-size, and muxes the original audio back in. Output mirrors <in-root>/<class>/<split>/*.mp4 under <out-root>.

Examples
--------
  # smoke-test: 3 clips/split from the first 2 classes, plus a preview montage
  python preprocess_mouth_roi.py --num-classes 2 --limit 3 --preview

  # the 15-class subset both failing models used, 6 parallel workers
  python preprocess_mouth_roi.py --num-classes 15 --jobs 6

  # everything
  python preprocess_mouth_roi.py --jobs 6
"""
import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

IN_ROOT_DEFAULT = './GLips/lipread_files'
OUT_ROOT_DEFAULT = './GLips_mouth/lipread_files'
SPLITS_DEFAULT = ('train', 'val', 'test')

import mediapipe as mp  # noqa: E402
LIP_IDX = sorted({i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair})

_FACE_MESH = None  # one per worker process, created lazily (not picklable across spawn)


def _face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,      # video mode: tracks between frames
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _FACE_MESH


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def lip_box(landmarks, w, h, box_scale):
    """Square (cx, cy, side) in pixels around the lips, or None if landmarks absent."""
    xs = [landmarks[i].x * w for i in LIP_IDX]
    ys = [landmarks[i].y * h for i in LIP_IDX]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * box_scale
    side = max(side, 24.0)  # guard against degenerate boxes from a bad detection
    return cx, cy, side


def smooth(prev, cur, alpha):
    if prev is None:
        return cur
    return tuple(alpha * c + (1 - alpha) * p for p, c in zip(prev, cur))


def crop_square(frame, box, out_size):
    """Crop square `box`=(cx,cy,side) from `frame` (replicate-padded if it runs off the edge), resized to out_size."""
    h, w = frame.shape[:2]
    cx, cy, side = box
    side = max(1, int(round(side)))
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = left + side, top + side

    pl, pt = max(0, -left), max(0, -top)
    pr, pb = max(0, right - w), max(0, bottom - h)
    bl, bt = max(0, left), max(0, top)
    br, bb = min(w, right), min(h, bottom)

    crop = frame[bt:bb, bl:br]
    if crop.size == 0:
        crop = frame
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)


def process_one(in_path, out_path, out_size, box_scale, alpha, overwrite):
    """Crop one clip to a mouth ROI and write it (with original audio) to out_path; returns per-clip stats dict."""
    if os.path.exists(out_path) and not overwrite:
        return {'path': in_path, 'status': 'skip'}

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return {'path': in_path, 'status': 'open_fail'}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0:
        fps = 25.0

    fm = _face_mesh()
    frames, n_detect, n_total = [], 0, 0
    last_box = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_total += 1
        h, w = frame.shape[:2]
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if res.multi_face_landmarks:
            n_detect += 1
            box = lip_box(res.multi_face_landmarks[0].landmark, w, h, box_scale)
            last_box = smooth(last_box, box, alpha)
        elif last_box is None:
            last_box = (w / 2, h / 2, min(w, h) * 0.4)  # no detection yet — centred fallback
        frames.append(crop_square(frame, last_box, out_size))
    cap.release()

    if not frames:
        return {'path': in_path, 'status': 'no_frames'}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_dir = os.path.dirname(out_path)
    fd, tmp_video = tempfile.mkstemp(suffix='.mp4', dir=tmp_dir)
    os.close(fd)
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (out_size, out_size))
    if not writer.isOpened():
        os.remove(tmp_video)
        return {'path': in_path, 'status': 'writer_fail'}
    for f in frames:
        writer.write(f)
    writer.release()

    cmd = [_ffmpeg_exe(), '-y', '-hide_banner', '-loglevel', 'error',
           '-i', tmp_video, '-i', in_path,
           '-map', '0:v:0', '-map', '1:a?',  # 1:a? tolerates clips with no audio
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
           '-shortest', '-movflags', '+faststart', out_path]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    os.remove(tmp_video)
    if rc != 0:
        return {'path': in_path, 'status': 'mux_fail'}

    return {'path': in_path, 'status': 'ok',
            'detect_rate': n_detect / n_total if n_total else 0.0, 'frames': n_total}


def _worker(args):
    return process_one(*args)


def gather_jobs(in_root, out_root, classes, splits, out_size, box_scale, alpha, overwrite):
    jobs = []
    for cls in classes:
        for split in splits:
            src = os.path.join(in_root, cls, split)
            if not os.path.isdir(src):
                continue
            for f in sorted(os.listdir(src)):
                if f.endswith('.mp4'):
                    jobs.append((os.path.join(src, f),
                                 os.path.join(out_root, cls, split, f),
                                 out_size, box_scale, alpha, overwrite))
    return jobs


def make_preview(jobs, out_size, box_scale, alpha, path, n=6):
    """Render raw-vs-mouth-crop middle frames for the first n jobs to a PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jobs = jobs[:n]
    fig, axes = plt.subplots(len(jobs), 2, figsize=(5, 2.4 * len(jobs)))
    axes = np.atleast_2d(axes)
    fm = _face_mesh()
    for r, (in_path, *_rest) in enumerate(jobs):
        cap = cv2.VideoCapture(in_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frames // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = frame.shape[:2]
        res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        box = (lip_box(res.multi_face_landmarks[0].landmark, w, h, box_scale)
               if res.multi_face_landmarks else (w / 2, h / 2, min(w, h) * 0.4))
        crop = crop_square(frame, box, out_size)
        axes[r, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cx, cy, side = box
        from matplotlib.patches import Rectangle
        axes[r, 0].add_patch(Rectangle((cx - side / 2, cy - side / 2), side, side,
                                       fill=False, edgecolor='lime', lw=2))
        axes[r, 0].set_title(os.path.basename(in_path), fontsize=7)
        axes[r, 1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axes[r, 1].set_title(f"mouth crop {out_size}x{out_size}", fontsize=8)
        for c in range(2):
            axes[r, c].axis('off')
    fig.suptitle("Mouth-ROI preview: green box = crop region", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=110)
    print(f"wrote preview {path}")


def first_classes(in_root, n):
    classes = sorted(d for d in os.listdir(in_root) if os.path.isdir(os.path.join(in_root, d)))
    return classes if n is None else classes[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in-root', default=IN_ROOT_DEFAULT)
    ap.add_argument('--out-root', default=OUT_ROOT_DEFAULT)
    ap.add_argument('--num-classes', type=int, default=None,
                    help='process only the first N alphabetical classes (default: all)')
    ap.add_argument('--classes', nargs='+', default=None,
                    help='explicit class/word folders to process (overrides --num-classes; '
                         'e.g. the 15 LRW words for the transfer experiment)')
    ap.add_argument('--splits', nargs='+', default=list(SPLITS_DEFAULT))
    ap.add_argument('--out-size', type=int, default=96)
    ap.add_argument('--box-scale', type=float, default=1.6,
                    help='square side = mouth_width * box_scale (more = wider context)')
    ap.add_argument('--alpha', type=float, default=0.4, help='EMA smoothing for the box (0-1)')
    ap.add_argument('--jobs', type=int, default=1, help='parallel worker processes')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap clips per (class,split) — for quick smoke tests')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--preview', action='store_true',
                    help='also write a raw-vs-crop montage to <out-root>/../mouth_preview.png')
    args = ap.parse_args()

    if not os.path.isdir(args.in_root):
        raise SystemExit(f"input root not found: {args.in_root} (run from the lipreading/ dir)")

    if args.classes is not None:
        missing = [c for c in args.classes
                   if not os.path.isdir(os.path.join(args.in_root, c))]
        if missing:
            raise SystemExit(f"--classes not found under {args.in_root}: {missing}")
        classes = args.classes
    else:
        classes = first_classes(args.in_root, args.num_classes)
    jobs = gather_jobs(args.in_root, args.out_root, classes, args.splits,
                       args.out_size, args.box_scale, args.alpha, args.overwrite)
    if args.limit is not None:
        seen, capped = {}, []  # keep the first `limit` per (class,split) bucket
        for j in jobs:
            bucket = os.path.dirname(j[0])
            seen[bucket] = seen.get(bucket, 0) + 1
            if seen[bucket] <= args.limit:
                capped.append(j)
        jobs = capped

    print(f"{len(classes)} classes, {len(jobs)} clips -> {args.out_root} "
          f"(out_size={args.out_size}, box_scale={args.box_scale}, jobs={args.jobs})")
    if not jobs:
        return

    if args.preview:
        preview_path = os.path.join(os.path.dirname(args.out_root.rstrip('/\\')), 'mouth_preview.png')
        os.makedirs(os.path.dirname(preview_path) or '.', exist_ok=True)
        make_preview(jobs, args.out_size, args.box_scale, args.alpha, preview_path)

    stats = {'ok': 0, 'skip': 0, 'low_detect': 0}
    fails = []
    low = []

    def tally(r):
        st = r['status']
        if st == 'ok':
            stats['ok'] += 1
            if r.get('detect_rate', 1.0) < 0.5:
                stats['low_detect'] += 1
                low.append((r['path'], r['detect_rate']))
        elif st == 'skip':
            stats['skip'] += 1
        else:
            fails.append((r['path'], st))

    done = 0
    if args.jobs <= 1:
        for j in jobs:
            tally(process_one(*j))
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} (ok={stats['ok']} skip={stats['skip']} "
                      f"fail={len(fails)} low_detect={stats['low_detect']})")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for fut in as_completed(futs):
                tally(fut.result())
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(jobs)} (ok={stats['ok']} skip={stats['skip']} "
                          f"fail={len(fails)} low_detect={stats['low_detect']})")

    print(f"\nDone. ok={stats['ok']} skip={stats['skip']} fail={len(fails)} "
          f"low_detect(<50% frames)={stats['low_detect']}")
    if low[:10]:
        print("Low-detection clips (check these — detector struggled):")
        for p, rate in low[:10]:
            print(f"  {rate:5.1%}  {p}")
    if fails[:10]:
        print("Failures:")
        for p, st in fails[:10]:
            print(f"  {st:12s} {p}")


if __name__ == '__main__':
    main()
