from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / filename
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
