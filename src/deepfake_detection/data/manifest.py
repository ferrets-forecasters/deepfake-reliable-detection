from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SegmentRecord:
    segment_id: str
    dataset: str
    subject_id: str
    video_path: str
    audio_path: str
    start_sec: float
    end_sec: float
    label: int  # 1 = fake, 0 = real
    split: str = ""  # filled in by splits.py after subject-level assignment


def write_manifest(records: list[SegmentRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in records]).to_csv(path, index=False)


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
