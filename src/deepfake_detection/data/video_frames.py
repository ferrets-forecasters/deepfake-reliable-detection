from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from deepfake_detection.data.face_crop import FaceBoxTracker, MultiFaceTracker


def _read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not read frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_video_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    face_tracker: FaceBoxTracker,
    image_processor,
) -> torch.Tensor:
    """Uniformly samples num_frames frames from [start_sec, end_sec), crops each
    to its interpolated face box, and runs them through the model's own image
    processor. Returns pixel_values shaped [num_frames, 3, image_size, image_size]."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).round().astype(int).tolist()

    frame_cache: dict[int, np.ndarray] = {}

    def reader(index: int) -> np.ndarray:
        if index not in frame_cache:
            frame_cache[index] = _read_frame(cap, index)
        return frame_cache[index]

    boxes = face_tracker.boxes_for_indices(reader, frame_indices, fps)

    cropped_frames = []
    for idx in frame_indices:
        frame = reader(idx)
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[idx])
        cropped_frames.append(frame[y0:y1, x0:x1])
    cap.release()

    processed = image_processor(cropped_frames, return_tensors="pt")
    return processed["pixel_values"][0]


def extract_multi_face_frames(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    num_frames: int,
    tracker: MultiFaceTracker,
    image_processor,
) -> list[torch.Tensor]:
    """Same sampling as extract_video_frames, but returns one pixel_values tensor
    [num_frames, 3, image_size, image_size] per face track, largest face first."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(round(start_sec * fps))
    end_frame = int(round(end_sec * fps))
    frame_indices = np.linspace(start_frame, end_frame - 1, num_frames).round().astype(int).tolist()

    frame_cache: dict[int, np.ndarray] = {}

    def reader(index: int) -> np.ndarray:
        if index not in frame_cache:
            frame_cache[index] = _read_frame(cap, index)
        return frame_cache[index]

    tracks = tracker.tracks_for_indices(reader, frame_indices, fps)
    per_track = []
    for track in tracks:
        boxes = track.boxes_for_indices(frame_indices)
        crops = []
        for idx in frame_indices:
            x0, y0, x1, y1 = (int(round(v)) for v in boxes[idx])
            crops.append(reader(idx)[y0:y1, x0:x1])
        per_track.append(image_processor(crops, return_tensors="pt")["pixel_values"][0])
    cap.release()
    return per_track
