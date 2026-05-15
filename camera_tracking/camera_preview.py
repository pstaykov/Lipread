import argparse
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from tempfile import NamedTemporaryFile

project_dir = Path(__file__).resolve().parent
venv_dir = project_dir / ".venv"
venv_python = project_dir / ".venv" / "bin" / "python"
if venv_python.exists() and Path(sys.prefix).resolve() != venv_dir.resolve():
    os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])

matplotlib_cache = project_dir / ".cache" / "matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(project_dir / ".cache"))

import cv2
import mediapipe as mp


NOSE_LANDMARKS = (
    1,
    2,
    4,
    5,
    6,
    19,
    94,
    97,
    98,
    168,
    195,
    197,
    326,
    327,
)


class FaceCropper:
    def __init__(
        self,
        min_detection_confidence=0.5,
        padding=0.01,
        head_scale=4.0,
        nose_y_position=0.45,
        smoothing_frames=0,
        output_size=256,
        max_faces=4,
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
        self.face_mesh = None
        self.face_detector = None
        self.max_faces = max(1, max_faces)
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.max_faces,
                refine_landmarks=False,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
            )
        except RuntimeError as error:
            print(
                f"MediaPipe face mesh unavailable; falling back to OpenCV tracking: {error}",
                file=sys.stderr,
            )
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(str(cascade_path))
            if self.face_detector.empty():
                raise RuntimeError(f"Could not load OpenCV face detector: {cascade_path}")

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()

    def process_frame(self, frame):
        if self.face_mesh is None:
            return self.process_frame_with_opencv(frame)

        image_height, image_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return self.crop_square(frame, self.last_target)

        targets = [
            self.target_from_landmarks(face_landmarks.landmark, image_width, image_height)
            for face_landmarks in results.multi_face_landmarks
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

    def target_from_landmarks(self, landmarks, image_width, image_height):
        nose_points = [
            (landmarks[index].x * image_width, landmarks[index].y * image_height)
            for index in NOSE_LANDMARKS
        ]

        nose_left = min(point[0] for point in nose_points)
        nose_right = max(point[0] for point in nose_points)
        nose_top = min(point[1] for point in nose_points)
        nose_bottom = max(point[1] for point in nose_points)
        nose_center_x = (nose_left + nose_right) / 2
        nose_center_y = (nose_top + nose_bottom) / 2

        nose_span = max(1, nose_right - nose_left, nose_bottom - nose_top)
        side = nose_span * self.head_scale * (1 + self.padding)
        target = (
            nose_center_x,
            nose_center_y + side * (0.5 - self.nose_y_position),
            side,
        )
        return target

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


def process_video(input_path, temp_video_path):
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
        head_scale=5.0,
        nose_y_position=0.45,
        smoothing_frames=1,
        output_size=OUTPUT_SIZE,
        max_faces=4,
    )

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

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
        print("ffmpeg is required but was not found on PATH.", file=sys.stderr)
        return 1

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
    args = parser.parse_args()

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
        process_result = process_video(input_path, temp_video_path)
        if process_result != 0:
            return process_result

        mux_result = mux_with_original_audio(temp_video_path, input_path, output_path)
        if mux_result != 0:
            return mux_result
    finally:
        temp_video_path.unlink(missing_ok=True)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
