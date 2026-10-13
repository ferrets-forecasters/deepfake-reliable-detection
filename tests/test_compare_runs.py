import importlib.util
from pathlib import Path

import numpy as np


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("compare_runs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(models_dir: Path, name: str, labels: np.ndarray, probs: np.ndarray) -> None:
    folder = models_dir / name
    folder.mkdir(parents=True)
    np.savez(folder / "alpha_primary.npz", alpha=np.zeros(len(labels), dtype=np.float32), labels=labels, probs=probs, manip_type=np.zeros(len(labels), dtype=np.int32))


def test_cross_validation_pools_every_fold_and_pairs_the_same_videos(tmp_path):
    rng = np.random.default_rng(0)
    module = _load()
    for fold in range(3):
        labels = rng.integers(0, 2, 200)
        strong = np.clip(labels * 0.8 + rng.normal(0.1, 0.1, 200), 0, 1)
        weak = np.clip(labels * 0.5 + rng.normal(0.25, 0.2, 200), 0, 1)
        _write(tmp_path, f"moe_cv_fold{fold}", labels, strong)
        _write(tmp_path, f"concat_cv_fold{fold}", labels, weak)
    result = module.cross_validation(tmp_path, "moe_cv_fold", "concat_cv_fold", 5, 50, 42)
    assert result["folds_done"] == [0, 1, 2]
    assert result["pooled"]["videos"] == 600
    assert result["pooled"]["gated_minus_static"]["auc"]["difference"] > 0
    assert "paired_t_p_value" in result["mean_sd"]["auc"]
