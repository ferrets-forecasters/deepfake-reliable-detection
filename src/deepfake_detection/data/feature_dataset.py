from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedFeatureDataset(Dataset):
    """Reads pre-extracted, frozen-encoder features cached by scripts/extract_features.py.

    Expects one manifest.npz per segment at
    features_dir/<dataset>/<segment_id>.npz, holding arrays "visual"
    ([T_v, 768]), "audio" ([T_a, 1024]) and "concat" ([1792], the z-scored
    concatenation used by the static baseline and the domain classifier).
    """

    def __init__(self, manifest: pd.DataFrame, features_dir: Path, domain_map: dict[str, int] | None = None):
        self.manifest = manifest.reset_index(drop=True)
        self.features_dir = features_dir
        self.domain_map = domain_map or {}

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[index]
        feature_path = self.features_dir / row["dataset"] / f"{row['segment_id']}.npz"
        cached = np.load(feature_path)

        item = {
            "visual_seq": torch.from_numpy(cached["visual"]).float(),
            "audio_seq": torch.from_numpy(cached["audio"]).float(),
            "concat_repr": torch.from_numpy(cached["concat"]).float(),
            "label": torch.tensor(float(row["label"])),
        }
        if self.domain_map:
            item["domain_label"] = torch.tensor(self.domain_map[row["dataset"]])
        return item
