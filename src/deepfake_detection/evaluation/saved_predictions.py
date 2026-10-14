from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_saved_predictions(predictions_dir: Path) -> dict[str, np.ndarray]:
    """probs, labels, alpha (nan where the model has no gate) and manip_type of one saved prediction folder:
    `predictions.npz` of the May models, `predictions.csv` (best-F1 checkpoint columns) of the retrained ones."""
    npz = predictions_dir / "predictions.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    with open(predictions_dir / "predictions.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "probs": np.array([float(r["p_fake_best_f1"]) for r in rows]),
        "labels": np.array([int(float(r["label"])) for r in rows]),
        "alpha": np.array([float(r["alpha_best_f1"]) if r.get("alpha_best_f1") not in (None, "") else np.nan for r in rows]),
        "manip_type": np.array([r["manip_type"] for r in rows]),
    }
