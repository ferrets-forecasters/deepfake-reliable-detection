from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from deepfake_detection.data.face_crop import DEFAULT_LANDMARKER_MODEL


class LandmarkMesh:
    """The process() interface of the legacy mediapipe FaceMesh (one face, 478 landmarks), on the Tasks FaceLandmarker of current mediapipe."""

    def __init__(self, model_path=None, min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path or DEFAULT_LANDMARKER_MODEL)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def process(self, frame_rgb: np.ndarray) -> SimpleNamespace:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        faces = self._landmarker.detect(image).face_landmarks
        return SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=faces[0])] if faces else None)
