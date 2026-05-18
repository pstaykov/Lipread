import argparse
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve

import numpy as np

project_dir = Path(__file__).resolve().parent
venv_dir = project_dir / ".venv"
venv_python = project_dir / ".venv" / "bin" / "python"
if venv_python.exists() and Path(sys.prefix).resolve() != venv_dir.resolve():
    os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])

matplotlib_cache = project_dir / ".cache" / "matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(project_dir / ".cache"))


def _add_cuda_dll_directories() -> None:
    if os.name != "nt":
        return

    dll_dirs = []
    for path_entry in sys.path:
        path = Path(path_entry)
        if not path.is_dir() or path.name != "site-packages":
            continue

        nvidia_root = path / "nvidia"
        if not nvidia_root.is_dir():
            continue

        for package_dir in nvidia_root.iterdir():
            bin_dir = package_dir / "bin"
            if bin_dir.is_dir() and any(bin_dir.glob("*.dll")):
                dll_dirs.append(bin_dir)

    for dll_dir in dll_dirs:
        os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, FileNotFoundError):
            pass


_add_cuda_dll_directories()

import cv2
try:
    import onnxruntime as ort
except ImportError:
    ort = None


YUNET_MODEL_NAME = "yunet.onnx"
YUNET_MODEL_URL = (
    "https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/"
    "a61a428929148171b488f024b5d6774f93cdbc13/tasks/task1/onnx/yunet.onnx"
)
YUNET_MODEL_PATH = project_dir / "models" / YUNET_MODEL_NAME


def ensure_yunet_model() -> Path:
    if YUNET_MODEL_PATH.is_file():
        return YUNET_MODEL_PATH

    YUNET_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(YUNET_MODEL_URL, YUNET_MODEL_PATH)
    except Exception as error:
        raise RuntimeError(f"Could not download YuNet model: {error}") from error
    return YUNET_MODEL_PATH


class YuNetDetector:
    def __init__(self, model_path, min_detection_confidence=0.9, prefer_gpu=True):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")

        available_providers = list(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        if prefer_gpu and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.nms_thresh = 0.3
        self.score_thresh = min_detection_confidence
        # The exported YuNet ONNX uses four feature levels with these strides
        # and these anchors-per-point (observed from the exported graph).
        self.strides = [8, 16, 32, 64]
        self.anchors_per_point = [3, 2, 2, 3]
        self.kps_count = 5
        self.center_cache = {}

    def close(self):
        pass

    def _anchor_centers(self, shape, stride):
        key = (shape[0], shape[1], stride)
        if key in self.center_cache:
            return self.center_cache[key]

        centers = np.stack(
            np.mgrid[: shape[0] // stride, : shape[1] // stride][::-1],
            axis=-1,
        )
        centers = (centers * stride).astype(np.float32).reshape(-1, 2)
        self.center_cache[key] = centers
        return centers

    @staticmethod
    def _nms(detections, threshold):
        if detections.size == 0:
            return []

        boxes = detections[:, :4].copy()
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        scores = detections[:, -1].tolist()
        keep = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores,
            score_threshold=0.0,
            nms_threshold=threshold,
            eta=1,
            top_k=5000,
        )
        if len(keep) == 0:
            return []
        return keep.flatten().tolist()

    def detect(self, image):
        image_height, image_width = image.shape[:2]
        blob = np.transpose(image, [2, 0, 1]).astype(np.float32)[np.newaxis, ...].copy()
        outputs = self.session.run(None, {self.input_name: blob})

        # ONNX export layout: loc (N,14), conf (N,2), iou (N,1)
        if len(outputs) < 3:
            return np.empty((0, 15), dtype=np.float32)

        loc = outputs[0]
        conf = outputs[1]
        iou = outputs[2]

        # Ensure shapes are (1, N, C) where appropriate
        if loc.ndim == 2:
            loc = loc.reshape(1, loc.shape[0], loc.shape[1])
        if conf.ndim == 2:
            conf = conf.reshape(1, conf.shape[0], conf.shape[1])
        if iou.ndim == 2:
            iou = iou.reshape(1, iou.shape[0], iou.shape[1])

        num_preds = loc.shape[1]

        # Build priors matching the exported graph (strides and anchors-per-point observed above).
        priors = []
        offset = 0.5
        for stride, a in zip(self.strides, self.anchors_per_point):
            h = image_height // stride
            w = image_width // stride
            xs = (np.arange(w) + offset) * stride
            ys = (np.arange(h) + offset) * stride
            cx, cy = np.meshgrid(xs, ys)
            cx = cx.ravel()
            cy = cy.ravel()
            for _ in range(a):
                priors.append(np.stack([cx, cy, np.full_like(cx, stride), np.full_like(cx, stride)], axis=1))

        if len(priors) == 0:
            return np.empty((0, 15), dtype=np.float32)

        priors = np.concatenate(priors, axis=0).astype(np.float32)
        if priors.shape[0] != num_preds:
            # fallback: try swapping H/W rounding or return empty
            return np.empty((0, 15), dtype=np.float32)

        # Decode boxes: first 4 loc entries are bbox (cx,cy,log(w),log(h) style)
        bbox_preds = loc[0, :, :4]
        centers = (bbox_preds[:, :2] * priors[:, 2:]) + priors[:, :2]
        wh = np.exp(bbox_preds[:, 2:4]) * priors[:, 2:]
        tl = centers - wh / 2.0
        br = centers + wh / 2.0
        boxes = np.hstack([tl, br])

        # Decode keypoints: remaining 10 values (5 xy pairs)
        kps_preds = loc[0, :, 4:4 + self.kps_count * 2]
        kps = []
        for i in range(self.kps_count):
            pair = (kps_preds[:, [2 * i, 2 * i + 1]] * priors[:, 2:]) + priors[:, :2]
            kps.append(pair)
        kps = np.concatenate(kps, axis=1)

        # Compute scores: use class probability (conf) and iou (sigmoid)
        cls_score = conf[0].max(axis=1)
        iou_score = 1.0 / (1.0 + np.exp(-iou[0].ravel()))
        scores = cls_score * iou_score

        score_mask = scores > self.score_thresh
        if not score_mask.any():
            return np.empty((0, 15), dtype=np.float32)

        boxes = boxes[score_mask]
        kps = kps[score_mask]
        scores = scores[score_mask]

        pre_det = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = self._nms(pre_det, self.nms_thresh)
        if not keep:
            return np.empty((0, 15), dtype=np.float32)

        pre_det = pre_det[keep, :]
        kps = kps[keep, :]
        return np.hstack((pre_det[:, :4], kps, pre_det[:, 4:5]))


class FaceCropper:
    def __init__(
        self,
        min_detection_confidence=0.9,
        padding=0.01,
        head_scale=1.6,
        nose_y_position=0.45,
        smoothing_frames=0,
        output_size=256,
        max_faces=4,
        prefer_gpu=True,
    ):
        self.padding = padding
        self.head_scale = head_scale
        self.nose_y_position = nose_y_position
        self.smoothing_frames = max(1, smoothing_frames)
        self.smoothing_alpha = 2 / (self.smoothing_frames + 1)
        self.output_size = max(1, output_size)
        self.tracks = []
        self.next_track_id = 1
        self.selected_track_id = None
        self.last_target = None
        self.face_detector = None
        self.max_faces = max(1, max_faces)
        self.prefer_gpu = prefer_gpu
        self.detector_mode = None
        self.model_path = ensure_yunet_model()

        try:
            self.face_detector = self.create_yunet_detector(
                model_path=self.model_path,
                min_detection_confidence=min_detection_confidence,
                prefer_gpu=self.prefer_gpu,
            )
            self.detector_mode = "yunet"
        except RuntimeError as error:
            print(
                f"YuNet detector unavailable; falling back to OpenCV Haar cascade: {error}",
                file=sys.stderr,
            )
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(str(cascade_path))
            if self.face_detector.empty():
                raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")
            self.detector_mode = "cascade"

    def close(self):
        pass

    def create_yunet_detector(self, model_path, min_detection_confidence, prefer_gpu):
        return YuNetDetector(
            model_path=model_path,
            min_detection_confidence=min_detection_confidence,
            prefer_gpu=prefer_gpu,
        )

    def process_frame(self, frame, timestamp_ms=None):
        if self.detector_mode != "yunet":
            return self.process_frame_with_opencv(frame)

        detections = self.face_detector.detect(frame)
        if detections is None or len(detections) == 0:
            return self.crop_square(frame, self.last_target)

        targets = [
            target
            for target in (
                self.target_from_detection(detection)
                for detection in detections[: self.max_faces]
            )
            if target is not None
        ]
        self.update_tracks(targets)
        selected_track = self.selected_track()
        if selected_track is None:
            return self.crop_square(frame, self.last_target)

        self.last_target = selected_track["smoothed_target"]
        return self.crop_square(frame, self.last_target)

    def process_frame_with_opencv(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        if len(faces) == 0:
            return self.crop_square(frame, self.last_target)

        targets = [
            self.target_from_face_box(face)
            for face in sorted(faces, key=lambda item: item[0])[: self.max_faces]
        ]
        self.update_tracks(targets)
        selected_track = self.selected_track()
        if selected_track is None:
            return self.crop_square(frame, self.last_target)

        self.last_target = selected_track["smoothed_target"]
        return self.crop_square(frame, self.last_target)

    def target_from_detection(self, detection):
        if detection is None:
            return None

        values = [float(value) for value in detection]
        if len(values) < 15:
            return None

        x, y, width, height = values[:4]
        nose_x, nose_y = values[8], values[9]
        side = max(width, height) * self.head_scale * (1 + self.padding)
        return (
            nose_x,
            nose_y + side * (0.5 - self.nose_y_position),
            side,
        )

    def target_from_face_box(self, face):
        x, y, width, height = [float(value) for value in face]
        side = max(width, height) * (1 + self.padding)
        return (x + width / 2, y + height / 2, side)

    def update_tracks(self, targets):
        unmatched_tracks = list(self.tracks)
        updated_tracks = []

        for target in sorted(targets, key=lambda item: item[0]):
            track = self.nearest_track(target, unmatched_tracks)
            if track is None:
                track = {
                    "id": self.next_track_id,
                    "recent_targets": deque(maxlen=self.smoothing_frames),
                    "smoothed_target": None,
                }
                self.next_track_id += 1
            else:
                unmatched_tracks.remove(track)

            track["smoothed_target"] = self.smooth_target(target, track)
            updated_tracks.append(track)

        self.tracks = sorted(updated_tracks, key=lambda item: item["smoothed_target"][0])
        visible_ids = {track["id"] for track in self.tracks}
        if self.selected_track_id not in visible_ids:
            self.selected_track_id = self.tracks[0]["id"] if self.tracks else None

    def nearest_track(self, target, tracks):
        if not tracks:
            return None

        max_distance = max(80, target[2])
        track = min(
            tracks,
            key=lambda item: self.target_distance(target, item["smoothed_target"]),
        )
        if self.target_distance(target, track["smoothed_target"]) > max_distance:
            return None
        return track

    @staticmethod
    def target_distance(target, smoothed_target):
        if smoothed_target is None:
            return 0
        return (
            (target[0] - smoothed_target[0]) ** 2
            + (target[1] - smoothed_target[1]) ** 2
        ) ** 0.5

    def selected_track(self):
        for track in self.tracks:
            if track["id"] == self.selected_track_id:
                return track
        return self.tracks[0] if self.tracks else None

    def cycle_target(self):
        if not self.tracks:
            return

        selected_index = 0
        for index, track in enumerate(self.tracks):
            if track["id"] == self.selected_track_id:
                selected_index = index
                break
        self.selected_track_id = self.tracks[(selected_index + 1) % len(self.tracks)]["id"]

    def crop_square(self, frame, target):
        image_height, image_width = frame.shape[:2]
        if target is None:
            side = min(image_width, image_height)
            target = (image_width / 2, image_height / 2, side)

        center_x, center_y, side = target
        side = max(1, int(round(side)))
        left = int(round(center_x - side / 2))
        top = int(round(center_y - side / 2))
        right = left + side
        bottom = top + side

        pad_left = max(0, -left)
        pad_top = max(0, -top)
        pad_right = max(0, right - image_width)
        pad_bottom = max(0, bottom - image_height)

        bounded_left = max(0, left)
        bounded_top = max(0, top)
        bounded_right = min(image_width, right)
        bounded_bottom = min(image_height, bottom)

        cropped = frame[bounded_top:bounded_bottom, bounded_left:bounded_right]
        if cropped.size == 0:
            cropped = frame

        if any((pad_left, pad_top, pad_right, pad_bottom)):
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )

        if cropped.shape[0] != side or cropped.shape[1] != side:
            cropped = cv2.resize(cropped, (side, side), interpolation=cv2.INTER_LINEAR)

        return cv2.resize(
            cropped,
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_LINEAR,
        )

    def smooth_target(self, target, track):
        track["recent_targets"].append(tuple(float(value) for value in target))
        buffered_target = tuple(
            sum(recent_target[index] for recent_target in track["recent_targets"])
            / len(track["recent_targets"])
            for index in range(3)
        )

        if track["smoothed_target"] is None:
            track["smoothed_target"] = buffered_target
        else:
            track["smoothed_target"] = tuple(
                self.smoothing_alpha * buffered_target[index]
                + (1 - self.smoothing_alpha) * track["smoothed_target"][index]
                for index in range(3)
            )

        return track["smoothed_target"]


OUTPUT_SIZE = 96
OUTPUT_SUFFIX = "_tracked.mp4"


def tracked_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}{OUTPUT_SUFFIX}")


def center_crop_frame(frame, crop_ratio):
    if crop_ratio >= 1.0:
        return frame

    image_height, image_width = frame.shape[:2]
    crop_width = max(1, int(round(image_width * crop_ratio)))
    crop_height = max(1, int(round(image_height * crop_ratio)))
    offset_x = max(0, (image_width - crop_width) // 2)
    offset_y = max(0, (image_height - crop_height) // 2)
    return frame[offset_y:offset_y + crop_height, offset_x:offset_x + crop_width]


def process_video(input_path, temp_video_path, center_crop_ratio):
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        print(f"Could not open video: {input_path}", file=sys.stderr)
        return 1

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(temp_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (OUTPUT_SIZE, OUTPUT_SIZE),
    )
    if not writer.isOpened():
        capture.release()
        print(f"Could not create temporary video: {temp_video_path}", file=sys.stderr)
        return 1

    cropper = FaceCropper(
        padding=0.12,
        head_scale=1.6,
        nose_y_position=0.45,
        smoothing_frames=1,
        output_size=OUTPUT_SIZE,
        max_faces=4,
        prefer_gpu=True,
    )

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame = center_crop_frame(frame, center_crop_ratio)
            writer.write(cropper.process_frame(frame))
            processed_count += 1
            if frame_count > 0 and processed_count % 30 == 0:
                print(f"Tracked {processed_count}/{frame_count} frames...", file=sys.stderr)
    finally:
        cropper.close()
        writer.release()
        capture.release()

    if processed_count == 0:
        print(f"No frames were read from: {input_path}", file=sys.stderr)
        return 1

    print(f"Tracked {processed_count} frames.", file=sys.stderr)
    return 0


def mux_with_original_audio(temp_video_path, input_path, output_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("ffmpeg not found; writing a silent cropped video.", file=sys.stderr)
        shutil.move(str(temp_video_path), str(output_path))
        return 0

    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temp_video_path),
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-vf",
        f"scale={OUTPUT_SIZE}:{OUTPUT_SIZE}:flags=lanczos",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        print("ffmpeg failed while creating the final MP4.", file=sys.stderr)
        return result.returncode
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Track a face in a video and write a 96x96 MP4 with the original audio."
    )
    parser.add_argument("input_video", type=Path, help="Input video file.")
    parser.add_argument(
        "--center-crop-ratio",
        type=float,
        default=1.0,
        help="Keep only the center portion of each frame before face tracking (0-1].",
    )
    parser.add_argument(
        "--prefer-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try to run MediaPipe face detection on GPU first and fall back to CPU if needed.",
    )
    args = parser.parse_args()

    if args.center_crop_ratio <= 0 or args.center_crop_ratio > 1.0:
        print("--center-crop-ratio must be in the range (0, 1].", file=sys.stderr)
        return 2

    input_path = args.input_video.expanduser().resolve()
    if not input_path.is_file():
        print(f"Input video does not exist: {input_path}", file=sys.stderr)
        return 1

    output_path = tracked_output_path(input_path)
    if output_path == input_path:
        print("Output path would overwrite the input video.", file=sys.stderr)
        return 1

    with NamedTemporaryFile(
        suffix=".mp4",
        prefix=f"{input_path.stem}_tracking_",
        dir=input_path.parent,
        delete=False,
    ) as temp_file:
        temp_video_path = Path(temp_file.name)

    try:
        process_result = process_video(input_path, temp_video_path, args.center_crop_ratio)
        if process_result != 0:
            return process_result

        mux_result = mux_with_original_audio(temp_video_path, input_path, output_path)
        if mux_result != 0:
            return mux_result
    finally:
        temp_video_path.unlink(missing_ok=True)

    try:
        print(output_path)
    except UnicodeEncodeError:
        # Some Windows code pages cannot render all filename characters.
        safe_output = str(output_path).encode("ascii", "backslashreplace").decode("ascii")
        print(safe_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
