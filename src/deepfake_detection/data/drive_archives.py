from __future__ import annotations

from pathlib import Path

import yaml

from deepfake_detection.config.config import REPO_ROOT

LOCAL_FILE = REPO_ROOT / "configs" / "drive_archives.local.yaml"


def load_drive_archives(path: Path | None = None, required: bool = True) -> dict[str, list[tuple[str, str, int | None]]]:
    """Drive file id, file name and size (or None) of each archive of each dataset, from the untracked drive_archives.local.yaml (an empty table when it is missing and required is False)."""
    path = path or LOCAL_FILE
    if not path.exists() and not required:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; copy configs/drive_archives.example.yaml to it and fill in the Drive file ids")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {dataset: [(entry["id"], entry["name"], entry.get("bytes")) for entry in entries] for dataset, entries in raw.items()}
