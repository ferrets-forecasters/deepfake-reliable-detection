from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class ReferenceRepo:
    scan: Callable
    subject_split: Callable
    subject_extractors: dict[str, Callable]
    build_model: Callable
    make_loader: Callable
    run_epoch: Callable
    focal_loss: type
    device: Any


def load_reference_repo(root: Path) -> ReferenceRepo:
    """Import the old research repo's modules without changing it. Its modules expect both the repo root and its src folder on sys.path."""
    for path in (root / "src", root):
        sys.path.insert(0, str(path))
    from src.data import dataset
    from src.models.factory import build_model
    from src.training.engine import run_epoch
    from src.training.losses import FocalLoss
    from src.training.trainer import _make_loader
    from configs.config import DEVICE

    return ReferenceRepo(
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
