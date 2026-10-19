from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Reference:
    scan: Callable
    subject_split: Callable
    subject_extractors: dict[str, Callable]
    build_model: Callable
    make_loader: Callable
    run_epoch: Callable
    focal_loss: type
    device: Any


def load_reference() -> Reference:
    """The data, model and training code that produced the paper's checkpoints. Imported on first use, after the dataset roots of reference.config are set."""
    from deepfake_detection.reference.config import DEVICE
    from deepfake_detection.reference.data import dataset
    from deepfake_detection.reference.models.factory import build_model
    from deepfake_detection.reference.training.engine import run_epoch
    from deepfake_detection.reference.training.losses import FocalLoss
    from deepfake_detection.reference.training.trainer import _make_loader

    return Reference(
        scan=dataset.scan,
        subject_split=dataset.subject_split,
        subject_extractors={
            "polyglotfake": dataset._extract_subject_polyglot,
            "hidf": dataset._extract_subject_hidf,
            "celebdfv2": dataset._extract_subject_celebdfv2,
            "fakeavceleb": dataset._extract_subject_fakeavceleb,
            "ffpp": dataset._extract_subject_ffpp,
        },
        build_model=build_model,
        make_loader=_make_loader,
        run_epoch=run_epoch,
        focal_loss=FocalLoss,
        device=DEVICE,
    )
