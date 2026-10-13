from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from deepfake_detection.config.config import _project_root

DEFAULT_LANDMARKER_MODEL = _project_root() / "external" / "mediapipe" / "face_landmarker.task"

Box = tuple[float, float, float, float]

# Face mesh indices: the outer lip ring, and the inner upper and lower lip points whose gap is the mouth opening.
LIPS_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
INNER_UPPER_LIP, INNER_LOWER_LIP = 13, 14


@dataclass(frozen=True)
class FaceDetection:
    box: Box
    mouth_box: Box
    mouth_open: float  # inner lip gap divided by the face height, unitless


class FaceLandmarkDetector:
    """Mediapipe face mesh through the Tasks API (the legacy mp.solutions API is gone in
    current mediapipe). The model file is the official face_landmarker.task, kept in external/mediapipe."""

    def __init__(self, max_faces: int, margin: float, model_path=None, min_confidence: float = 0.5, detect_max_side: int = 480):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path or DEFAULT_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        self.margin = margin
        self.detect_max_side = detect_max_side

    def detect(self, frame_rgb: np.ndarray) -> list[FaceDetection]:
        height, width = frame_rgb.shape[:2]
        scale = self.detect_max_side / max(height, width)
        small = cv2.resize(frame_rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame_rgb
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        detections = []
        for face in self._landmarker.detect(image).face_landmarks:
            xs = np.array([lm.x for lm in face]) * width
            ys = np.array([lm.y for lm in face]) * height
            x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
            box_w, box_h = x_max - x_min, y_max - y_min
            box = (
                max(x_min - box_w * self.margin, 0),
                max(y_min - box_h * self.margin, 0),
                min(x_max + box_w * self.margin, width),
                min(y_max + box_h * self.margin, height),
            )
            lip_x, lip_y = xs[list(LIPS_OUTER)], ys[list(LIPS_OUTER)]
            pad_x, pad_y = (lip_x.max() - lip_x.min()) * 0.2, (lip_y.max() - lip_y.min()) * 0.2
            mouth_box = (
                max(lip_x.min() - pad_x, 0), max(lip_y.min() - pad_y, 0),
                min(lip_x.max() + pad_x, width), min(lip_y.max() + pad_y, height),
            )
            gap = abs(ys[INNER_LOWER_LIP] - ys[INNER_UPPER_LIP]) / max(box_h, 1e-6)
            detections.append(FaceDetection(box=box, mouth_box=mouth_box, mouth_open=float(gap)))
        return detections

    def boxes(self, frame_rgb: np.ndarray) -> list[Box]:
        return [d.box for d in self.detect(frame_rgb)]


class FaceBoxTracker:
    """Detects a face bounding box at a low FPS with Mediapipe FaceMesh and
    linearly interpolates it across every frame index the caller actually needs,
    instead of running detection on every frame."""

    def __init__(self, detector_fps: int, margin: float = 0.25, model_path=None):
        self._detector = FaceLandmarkDetector(max_faces=1, margin=margin, model_path=model_path)
        self.detector_fps = detector_fps
        self.margin = margin

    def _detect_box(self, frame_rgb: np.ndarray) -> Box | None:
        boxes = self._detector.boxes(frame_rgb)
        return boxes[0] if boxes else None

    def boxes_for_indices(self, frame_reader, frame_indices: list[int], video_fps: float) -> dict[int, Box]:
        """frame_reader(index) -> RGB frame array. Detects on frames spaced
        video_fps/detector_fps apart, interpolates the rest linearly."""
        stride = max(int(round(video_fps / self.detector_fps)), 1)
        first, last = frame_indices[0], frame_indices[-1]
        detection_indices = list(range(first, last + 1, stride))
        if detection_indices[-1] != last:
            detection_indices.append(last)

        detected: dict[int, Box] = {}
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


def _interpolate(known: dict[int, np.ndarray | float], indices: list[int]) -> dict[int, np.ndarray | float]:
    keys = sorted(known)
    values = np.array([known[k] for k in keys], dtype=float)
    if values.ndim == 1:
        return {idx: float(np.interp(idx, keys, values)) for idx in indices}
    return {idx: tuple(np.interp(idx, keys, values[:, dim]) for dim in range(values.shape[1])) for idx in indices}


@dataclass
class FaceTrack:
    track_id: int
    detections: dict[int, Box] = field(default_factory=dict)
    mouth_boxes: dict[int, Box] = field(default_factory=dict)
    mouth_open: dict[int, float] = field(default_factory=dict)

    @property
    def mean_area(self) -> float:
        return float(np.mean([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in self.detections.values()]))

    def center_x(self) -> float:
        return float(np.mean([(b[0] + b[2]) / 2 for b in self.detections.values()]))

    def boxes_for_indices(self, frame_indices: list[int]) -> dict[int, Box]:
        return _interpolate(self.detections, frame_indices)

    def mouth_boxes_for_indices(self, frame_indices: list[int]) -> dict[int, Box]:
        return _interpolate(self.mouth_boxes, frame_indices)

    def mouth_open_for_indices(self, frame_indices: list[int]) -> dict[int, float]:
        return _interpolate(self.mouth_open, frame_indices)

    def present_between(self, start_frame: int, end_frame: int, tolerance: int = 0) -> bool:
        """True if the track has a detection inside [start_frame - tolerance, end_frame + tolerance]."""
        return any(start_frame - tolerance <= idx <= end_frame + tolerance for idx in self.detections)


def box_iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def link_boxes(detections: dict[int, list[Box]], iou_min: float = 0.3) -> list[FaceTrack]:
    """Links per-frame face boxes into tracks, greedily by highest IoU with the
    last box of each open track. A face that is not matched starts a new track,
    a track that gets no match in a frame keeps its last box and waits."""
    tracks: list[FaceTrack] = []
    for idx in sorted(detections):
        pairs = sorted(
            (
                (box_iou(max(track.detections.items())[1], box), ti, bi)
                for ti, track in enumerate(tracks)
                for bi, box in enumerate(detections[idx])
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for score, ti, bi in pairs:
            if score < iou_min:
                break
            if ti in used_tracks or bi in used_boxes:
                continue
            tracks[ti].detections[idx] = detections[idx][bi]
            used_tracks.add(ti)
            used_boxes.add(bi)
        for bi, box in enumerate(detections[idx]):
            if bi not in used_boxes:
                tracks.append(FaceTrack(track_id=len(tracks), detections={idx: box}))
    return tracks


class MultiFaceTracker:
    """Detects every face (up to max_faces) at a low FPS with Mediapipe FaceMesh,
    links the boxes into tracks, and interpolates each track over the frames the
    caller needs. Tracks seen in fewer than min_hits detection frames are dropped
    as false positives; the rest come back ordered by mean face area, largest first.
    Each track also carries the mouth box and mouth opening at its detection frames."""

    def __init__(
        self, detector_fps: int, max_faces: int = 5, margin: float = 0.25, min_hits: int = 2, iou_min: float = 0.3, model_path=None
    ):
        self._detector = FaceLandmarkDetector(max_faces=max_faces, margin=margin, model_path=model_path)
        self.detector_fps = detector_fps
        self.min_hits = min_hits
        self.iou_min = iou_min

    def tracks_for_indices(self, frame_reader, frame_indices: list[int], video_fps: float) -> list[FaceTrack]:
        """Detection runs on every video_fps/detector_fps-th frame between the first and the last index."""
        stride = max(int(round(video_fps / self.detector_fps)), 1)
        first, last = frame_indices[0], frame_indices[-1]
        detection_indices = list(range(first, last + 1, stride))
        if detection_indices[-1] != last:
            detection_indices.append(last)

        found = {idx: self._detector.detect(frame_reader(idx)) for idx in detection_indices}
        lookup = {(idx, d.box): d for idx, faces in found.items() for d in faces}
        tracks = [
            t for t in link_boxes({idx: [d.box for d in faces] for idx, faces in found.items()}, self.iou_min)
            if len(t.detections) >= self.min_hits
        ]
        if not tracks:
            raise ValueError("no face track in this segment")
        for track in tracks:
            for idx, box in track.detections.items():
                track.mouth_boxes[idx] = lookup[(idx, box)].mouth_box
                track.mouth_open[idx] = lookup[(idx, box)].mouth_open
        return sorted(tracks, key=lambda t: t.mean_area, reverse=True)
