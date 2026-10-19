from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from deepfake_detection.config import Config
from deepfake_detection.reference import Reference

COMPARED_METRICS = ("f1", "ap", "auc", "alpha_mean")
PER_TYPE_KEYS = ("audio_video", "video_only")


def dataset_records(reference: Reference, config: Config, name: str) -> list:
    spec = config.dataset(name)
    return reference.scan(
        config.embedding_dir(name),
        reference.subject_extractors[spec.subject_rule],
        fake_type=spec.fake_type,
        audio_avail=spec.audio_available,
    )


def test_subjects_70_30(records: list, seed: int, test_share: float = 0.30) -> list:
    """The test part of the reference 56/14/30 subject split: shuffle the sorted subjects, the first 30% are the test."""
    rng = random.Random(seed)
    subjects = sorted({r[2] for r in records})
    rng.shuffle(subjects)
    test_subjects = set(subjects[: max(1, int(len(subjects) * test_share))])
    return [r for r in records if r[2] in test_subjects]


def test_records(reference: Reference, config: Config, experiment: dict, split: dict) -> list:
    if experiment["experiment"] == "leave_out":
        return dataset_records(reference, config, experiment["held_out"])
    if experiment["experiment"] == "split_70_30":
        return test_subjects_70_30(dataset_records(reference, config, experiment["dataset"]), split["seed"])
    records: list = []
    for name in config.dataset_names():
        _, _, test = reference.subject_split(dataset_records(reference, config, name), split["val_ratio"], split["seed"])
        records += test
    return records


def _resize_domain_head(model: torch.nn.Module, state: dict) -> None:
    key = "domain_head.3.weight"
    head = getattr(model, "domain_head", None)
    if key not in state or not isinstance(head, torch.nn.Sequential):
        return
    out_features, in_features = state[key].shape
    if head[-1].out_features != out_features:
        head[-1] = torch.nn.Linear(in_features, out_features)


def evaluate_model(reference: Reference, model_dir: Path, records: list, loader_cfg: dict, checkpoint: str = "best_f1") -> dict:
    meta = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))
    model = reference.build_model(meta["model"]).to(reference.device)
    state = torch.load(model_dir / f"{checkpoint}.pt", map_location=reference.device, weights_only=True)
    _resize_domain_head(model, state)
    model.to(reference.device).load_state_dict(state)
    stats = torch.load(model_dir / "norm_stats.pt", map_location="cpu", weights_only=False)["train_stats"]
    performance = {"num_workers": loader_cfg["num_workers"], "persistent_workers": False}
    loader, _ = reference.make_loader(records, stats, loader_cfg["batch_size"], shuffle=False, performance=performance)
    criterion = reference.focal_loss(gamma=1.0, alpha=0.75, label_smoothing=0.05)
    return reference.run_epoch(
        model, loader, criterion, None, is_training=False, desc=model_dir.name,
        amp_enabled=loader_cfg["amp"], amp_dtype=torch.float16,
    )


def compare_with_stored(result: dict, stored: dict, tolerance: float) -> dict:
    rows = {}
    for key in COMPARED_METRICS:
        expected, measured = stored.get(key), result.get(key)
        if expected is None or measured is None:
            rows[key] = {"expected": expected, "measured": measured, "delta": None, "ok": expected is None and measured is None}
            continue
        delta = float(measured) - float(expected)
        rows[key] = {"expected": expected, "measured": round(float(measured), 4), "delta": round(delta, 4), "ok": abs(delta) <= tolerance}
    for key in PER_TYPE_KEYS:
        expected, measured = stored.get("per_type_f1", {}).get(key), result["per_type_f1"].get(key)
        if expected is None or measured is None:
            continue
        delta = float(measured) - float(expected)
        rows[f"f1_{key}"] = {"expected": expected, "measured": round(float(measured), 4), "delta": round(delta, 4), "ok": abs(delta) <= tolerance}
    return rows


def confusion_counts(result: dict) -> dict:
    labels, preds = np.asarray(result["labels"]), np.asarray(result["preds"])
    return {
        "real_flagged_fake": int(((labels == 0) & (preds == 1)).sum()),
        "fake_missed": int(((labels == 1) & (preds == 0)).sum()),
        "real_total": int((labels == 0).sum()),
        "fake_total": int((labels == 1).sum()),
    }


def save_outputs(config: Config, model_name: str, scope: str, result: dict, comparison: dict, seconds: float) -> None:
    metrics_dir = config.paths.results_dir(f"reproduce_{model_name}")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model_name,
        "scope": scope,
        "test_items": len(result["labels"]),
        "seconds": round(seconds, 1),
        "all_within_tolerance": all(row["ok"] for row in comparison.values()),
        "errors_at_threshold_0.5": confusion_counts(result),
        "comparison": comparison,
    }
    (metrics_dir / "metrics.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    predictions_dir = config.paths.predictions_dir(f"{model_name}_on_{scope}")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    alpha = result["alpha_arr"] if result["alpha_arr"] is not None else np.zeros(len(result["labels_arr"]), dtype=np.float32)
    np.savez_compressed(
        predictions_dir / "predictions.npz",
        alpha=alpha, labels=result["labels_arr"], probs=result["probs_arr"], manip_type=result["manip_arr"],
    )


def timed(fn, *args, **kwargs):
    start = time.time()
    value = fn(*args, **kwargs)
    return value, time.time() - start
