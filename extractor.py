# extractor.py
# opens each video, detects scene changes, extracts meaningful frames
# stores frame_number, timestamp, image, and video_path with each frame

import cv2  # type: ignore
import numpy as np  # type: ignore

# threshold controls sensitivity of scene change detection
# higher = fewer frames (fast mode)
# lower  = more frames (better mode)
THRESHOLDS = {
    "fast":   30,
    "better": 8
}

# how often to sample frames (in seconds)
# fast mode samples less frequently for speed
# better mode samples more frequently for accuracy
SAMPLE_INTERVALS = {
    "fast":   1.0,    # check 1 frame per second
    "better": 0.5     # check 2 frames per second
}


def extract_frames(video_path, mode="better"):
    """
    Extract meaningful frames from a video using scene change detection.
    Returns a list of dicts with frame_number, timestamp, seconds, image, video_path.
    """
    threshold = THRESHOLDS.get(mode, THRESHOLDS["better"])
    sample_interval = SAMPLE_INTERVALS.get(mode, SAMPLE_INTERVALS["better"])

    try:
        video = cv2.VideoCapture(str(video_path))
    except Exception as e:
        print(f"  ERROR: Could not open video -> {video_path} ({e})")
        return []

    if not video.isOpened():
        print(f"  ERROR: Could not open video -> {video_path}")
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # guard against invalid fps
    if fps <= 0:
        print(f"  ERROR: Invalid FPS ({fps}) for -> {video_path}")
        video.release()
        return []

    duration = total_frames / fps
    print(f"  {video_path.name} | {duration:.1f}s | {total_frames} frames")

    # calculate how many frames to skip between samples
    frame_skip = max(1, int(fps * sample_interval))

    frames = []
    prev_gray = None
    frame_number = 0

    while True:
        # seek to the next sample position instead of reading every frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = video.read()
        if not success or frame is None:
            break

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            # skip corrupt / unreadable frames
            frame_number += frame_skip
            continue

        should_keep = False

        if prev_gray is None:
            # always keep the very first frame so we don't miss
            # content at the start of the video
            should_keep = True
        else:
            # scene change detection
            diff  = cv2.absdiff(gray, prev_gray)
            score = np.mean(diff)

            if score > threshold:
                should_keep = True

        if should_keep:
            # convert seconds into mm:ss string for display
            seconds   = frame_number / fps
            minutes   = int(seconds // 60)
            secs      = int(seconds % 60)
            timestamp = f"{minutes}:{secs:02d}"

            frames.append({
                "frame_number": frame_number,
                "timestamp":    timestamp,
                "seconds":      round(seconds, 2),
                "image":        frame,
                "video_path":   str(video_path)
            })

        prev_gray    = gray
        frame_number += frame_skip

        # safety: don't go past the video
        if frame_number >= total_frames:
            break

    video.release()

    print(f"    -> Extracted {len(frames)} frames (mode: {mode})")
    return frames