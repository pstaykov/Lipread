"""Image inputs of docs/MODELS.md: RGB crops (V0), mesh overlay (V1) and rendered mesh on black (V2, V3).

Crop boxes follow the chosen points frame by frame (smoothed centre, like the BWKI mouth crop) but keep one
size per clip, so the head can move (speakers sway ~75 px in 2 s) while mouth opening is not scaled away.
Meshes are drawn on a CANVAS-sized upscaled crop and then shrunk to the encoder size: at native ROI
resolution the face is ~110 px and the lines would merge.
"""
import cv2
import numpy as np

ENC_SIZE = 128     # NASNet input, as in Ameer et al.
CANVAS = 256       # drawing resolution
LINE_PX = 2        # on the canvas -> ~1 px at ENC_SIZE
FACE_MARGIN = 0.15
MOUTH_MARGIN = 0.6
SMOOTH = 5         # frames, moving average of the box centre


def clip_boxes(seq_px, idx, margin):
    """Per-frame square boxes (T, 3) as (x0, y0, side) in ROI pixels for points idx of seq_px (T, 468, 3):
    centre = smoothed bbox centre of the points, side = largest bbox side in the clip × (1 + margin)."""
    pts = seq_px[:, idx, :2]
    lo, hi = pts.min(1), pts.max(1)
    centre = (lo + hi) / 2
    pad = SMOOTH // 2
    padded = np.pad(centre, ((pad, pad), (0, 0)), mode='edge')
    centre = np.stack([padded[t:t + SMOOTH].mean(0) for t in range(len(centre))])
    side = (hi - lo).max() * (1 + margin)
    return np.column_stack([centre - side / 2, np.full(len(centre), side)])


def _to_canvas(pts, box, size):
    x0, y0, side = box
    return (pts[:, :2] - (x0, y0)) * (size / side)


def crop(frame, box, size):
    """Crop box from frame and resize to size × size; parts outside the ROI are black."""
    x0, y0, side = box
    s = size / side
    m = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float32)
    return cv2.warpAffine(frame, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)  # crops are upscaled


def draw(canvas, pts, edges):
    """Connect the dots: one anti-aliased white line per FaceMesh edge, in place."""
    p = np.round(pts).astype(int)
    for a, b in edges:
        cv2.line(canvas, tuple(p[a]), tuple(p[b]), (255, 255, 255), LINE_PX, cv2.LINE_AA)
    return canvas


def rgb_clip(frames, boxes, size=ENC_SIZE):
    """V0: (T, H, W, 3) uint8 ROI frames -> (T, size, size, 3) crops."""
    return np.stack([crop(f, box, size) for f, box in zip(frames, boxes)])


def overlay_clip(frames, seq_px, edges, boxes, size=ENC_SIZE):
    """V1: mesh drawn on the RGB crop at 100 % opacity."""
    out = []
    for f, pts, box in zip(frames, seq_px, boxes):
        c = draw(crop(f, box, CANVAS), _to_canvas(pts, box, CANVAS), edges)
        out.append(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def render_clip(seq_px, edges, boxes, size=ENC_SIZE):
    """V2 / V3: mesh only, on black, no video pixels."""
    out = []
    for pts, box in zip(seq_px, boxes):
        c = draw(np.zeros((CANVAS, CANVAS, 3), np.uint8), _to_canvas(pts, box, CANVAS), edges)
        out.append(cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(out)
