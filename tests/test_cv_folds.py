import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_cv_folds.py"
    spec = importlib.util.spec_from_file_location("make_cv_folds", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_folds_keep_every_subject_in_one_fold_and_balance_the_videos():
    assign_folds = _load().assign_folds
    records = [([f"{s}_{i}"], i % 4 == 0, f"subject{s}", "x", True) for s in range(40) for i in range(1 + s % 5)]
    folds = assign_folds(records, 5, 42)
    assert set(folds) == {f"subject{s}" for s in range(40)}
    sizes = [sum(1 for r in records if folds[r[2]] == f) for f in range(5)]
    assert max(sizes) - min(sizes) <= 3
    assert assign_folds(records, 5, 42) == folds
