from __future__ import annotations

import numpy as np


class FaceBoxTracker:
    """Detects a face bounding box at a low FPS with Mediapipe FaceMesh and
    linearly interpolates it across every frame index the caller actually needs,
    instead of running detection on every frame."""

    def __init__(self, detector_fps: int, margin: float = 0.25):
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )
        self.detector_fps = detector_fps
        self.margin = margin

    def _detect_box(self, frame_rgb: np.ndarray) -> tuple[float, float, float, float] | None:
        height, width = frame_rgb.shape[:2]
        result = self._face_mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None
        xs = [lm.x for lm in result.multi_face_landmarks[0].landmark]
        ys = [lm.y for lm in result.multi_face_landmarks[0].landmark]
        x_min, x_max = min(xs) * width, max(xs) * width
        y_min, y_max = min(ys) * height, max(ys) * height
        box_w, box_h = x_max - x_min, y_max - y_min
        x_min -= box_w * self.margin
        x_max += box_w * self.margin
        y_min -= box_h * self.margin
        y_max += box_h * self.margin
        return (max(x_min, 0), max(y_min, 0), min(x_max, width), min(y_max, height))

    def boxes_for_indices(
        self, frame_reader, frame_indices: list[int], video_fps: float
    ) -> dict[int, tuple[float, float, float, float]]:
        """frame_reader(index) -> RGB frame array. Detects on frames spaced
        video_fps/detector_fps apart, interpolates the rest linearly."""
        stride = max(int(round(video_fps / self.detector_fps)), 1)
        first, last = frame_indices[0], frame_indices[-1]
        detection_indices = list(range(first, last + 1, stride))
        if detection_indices[-1] != last:
            detection_indices.append(last)

        detected: dict[int, tuple[float, float, float, float]] = {}
        for idx in detection_indices:
            box = self._detect_box(frame_reader(idx))
            if box is not None:
                detected[idx] = box

        if not detected:
            raise ValueError("no face detected in this segment")

        known_indices = sorted(detected)
        boxes = np.array([detected[i] for i in known_indices])
        interpolated = {}
        for idx in frame_indices:
            interpolated[idx] = tuple(np.interp(idx, known_indices, boxes[:, dim]) for dim in range(4))
        return interpolated
