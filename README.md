# YouTube Channel Downloader

This script downloads a fixed number of videos from a YouTube channel, user, handle, or uploads page into a target folder.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Usage

```powershell
python download_channel_videos.py <source_url> <count> <output_dir>
```

Videos are downloaded at up to 720p.

Examples:

```powershell
python download_channel_videos.py https://www.youtube.com/@SomeChannel/videos 10 D:\Videos\SomeChannel
python download_channel_videos.py https://www.youtube.com/channel/CHANNEL_ID/videos 25 .\downloads
```

Options:

```powershell
--oldest-first
--use-cropper
--cropper <path_to_camera_preview.py>
--center-crop-ratio <float>
--prefer-gpu / --no-prefer-gpu
```

Use `--oldest-first` if you want the earliest videos from the channel instead of the newest ones.

Use `--center-crop-ratio` to keep only the center of each frame before face tracking (default: `0.6`). Set it to `1.0` to disable this pre-crop.
If you set `--center-crop-ratio` below `1.0`, the cropper will be used automatically when the script is available.

The cropper now uses the YuNet face detector model from the original libfacedetection.train source: `yunet.onnx`. It tries OpenCV's CUDA backend first, then falls back to CPU. If neither YuNet backend works, it falls back to OpenCV's Haar cascade detector.

Example with face cropper and center pre-crop:

```powershell
python download_channel_videos.py https://www.youtube.com/@SomeChannel/videos 10 .\output_folder --use-cropper --center-crop-ratio 0.6
```

## Notes

- If yt-dlp asks for browser cookies or age verification on a particular channel, you may need to provide cookies manually.
- Downloading best-quality video may require `ffmpeg` to be installed on your system.
