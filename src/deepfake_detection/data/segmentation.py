from __future__ import annotations

from pathlib import Path

import cv2


def video_duration_seconds(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"could not read fps for {video_path}")
    return frame_count / fps


def segment_windows(duration_sec: float, segment_seconds: int) -> list[tuple[float, float]]:
    """Non-overlapping windows; a trailing remainder shorter than segment_seconds is dropped."""
    num_segments = int(duration_sec // segment_seconds)
    return [(i * segment_seconds, (i + 1) * segment_seconds) for i in range(num_segments)]
