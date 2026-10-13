from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from deepfake_detection.config import DatasetSpec
from deepfake_detection.data.subject_rules import subject_id

LABEL_FOLDERS = {"real": 0, "fake": 1}


def build_embedding_manifest(spec: DatasetSpec, embedding_dir: Path, columns: list[str]) -> pd.DataFrame:
    """One row per sample, built from the segment files under embedding_dir/<real|fake>/<sample_id>_seg<NNN>.npz."""
    if len(spec.media_types) != 1:
        raise ValueError(f"{spec.name}: media type per sample needs the raw files when a dataset mixes {spec.media_types}")
    rows = []
    for folder, label in LABEL_FOLDERS.items():
        counts: Counter[str] = Counter()
        with os.scandir(embedding_dir / folder) as entries:
            for entry in tqdm(entries, desc=f"{spec.name}/{folder}", unit="file", leave=False):
                if entry.name.endswith(".npz") and "_seg" in entry.name:
                    counts[entry.name[: -len(".npz")].rsplit("_seg", 1)[0]] += 1
        for sample_id in sorted(counts):
            rows.append(
                {
                    "sample_id": sample_id,
                    "media_type": spec.media_types[0],
                    "raw_path": "",
                    "embedding_pattern": f"{folder}/{sample_id}_seg*.npz",
                    "num_segments": counts[sample_id],
                    "label": label,
                    "manipulation_type": "real" if label == 0 else spec.fake_type,
                    "subject_id": subject_id(spec.subject_rule, sample_id),
                    "audio_available": int(spec.audio_available),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def manifest_totals(manifest: pd.DataFrame) -> dict[str, int]:
    return {
        "real": int(manifest.loc[manifest["label"] == 0, "num_segments"].sum()),
        "fake": int(manifest.loc[manifest["label"] == 1, "num_segments"].sum()),
        "segments": int(manifest["num_segments"].sum()),
    }
