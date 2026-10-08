from __future__ import annotations

import numpy as np

from deepfake_detection.config import SplitConfig


def subject_level_split(subject_ids: list[str], split: SplitConfig, seed: int) -> dict[str, str]:
    """Assigns each unique subject to train/val/test so no identity crosses a split boundary."""
    unique_subjects = sorted(set(subject_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_train = int(round(n * split.train))
    n_val = int(round(n * split.val))

    assignment: dict[str, str] = {}
    for subject in unique_subjects[:n_train]:
        assignment[subject] = "train"
    for subject in unique_subjects[n_train : n_train + n_val]:
        assignment[subject] = "val"
    for subject in unique_subjects[n_train + n_val :]:
        assignment[subject] = "test"
    return assignment
