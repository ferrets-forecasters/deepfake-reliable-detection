from __future__ import annotations

from pathlib import Path

import yaml

from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference.multi_subject import MultiSubjectInference
from deepfake_detection.inference.speakers import SingleSpeaker
from deepfake_detection.reference import load_reference_repo


def build_pipeline(config, model: str, diarizer: str = "ecapa", detector_fps: float = 8.0, max_faces: int = 5, threshold: float = 0.5,
                   reproduction: Path | None = None) -> MultiSubjectInference:
    """The multi-person pipeline with an exported checkpoint: face tracks, per-speaker audio (diarizer 'ecapa') or the shared audio ('single')."""
    repro = yaml.safe_load((reproduction or REPO_ROOT / "configs" / "reproduction.yaml").read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    if diarizer == "ecapa":
        from deepfake_detection.inference.diarization import EcapaDiarizer

        speakers = EcapaDiarizer()
    else:
        speakers = SingleSpeaker()
    return MultiSubjectInference(
        tracker=MultiFaceTracker(detector_fps, max_faces=max_faces), embedder=SegmentEmbedder(),
        scorer=CheckpointScorer(reference, config.paths.checkpoints_dir(model)), diarizer=speakers, threshold=threshold,
    )
