from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepfake_detection.config import load_config

# scope of the baseline -> (our run folders, dataset column filter or None)
OURS = {
    "celebdfv2_official_test": {
        "gated, FF++ four types": "moe_ffpp_to_celebdfv2_retrain_on_celebdfv2_from_ffpp_four_types",
        "static, FF++ four types": "concat_ffpp_to_celebdfv2_retrain_on_celebdfv2_from_ffpp_four_types",
        "gated, FF++ five types": "moe_ffpp5_to_celebdfv2_retrain_on_celebdfv2_from_ffpp_five_types",
        "static, FF++ five types": "concat_ffpp5_to_celebdfv2_retrain_on_celebdfv2_from_ffpp_five_types",
    },
    "own_test_hidf": {"gated, combined training": "moe_combined_retrain_on_combined_test_seed42", "static, combined training": "concat_combined_retrain_on_combined_test_seed42"},
    "own_test_fakeavceleb": {"gated, combined training": "moe_combined_retrain_on_combined_test_seed42", "static, combined training": "concat_combined_retrain_on_combined_test_seed42"},
}
DATASET_OF_SCOPE = {"celebdfv2_official_test": "celebdfv2", "own_test_hidf": "hidf", "own_test_fakeavceleb": "fakeavceleb"}


def read_ours(path: Path, dataset: str) -> dict[str, float]:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset:
                out[re.sub(r"_seg\d+$", "", Path(row["video"]).stem)] = float(row["p_fake_best_f1"])
    return out


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(labels, scores)), "ap": float(average_precision_score(labels, scores)), "macro_f1_at_0.5": float(f1_score(labels, scores >= 0.5, average="macro"))}


def bootstrap(labels: np.ndarray, a: np.ndarray, b: np.ndarray, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    auc_a, auc_b = [], []
    for _ in range(resamples):
        pick = rng.integers(0, len(labels), len(labels))
        if len(set(labels[pick])) < 2:
            continue
        auc_a.append(roc_auc_score(labels[pick], a[pick]))
        auc_b.append(roc_auc_score(labels[pick], b[pick]))
    diff = np.array(auc_a) - np.array(auc_b)
    return {"auc_ci95": [round(float(np.quantile(auc_a, 0.025)), 4), round(float(np.quantile(auc_a, 0.975)), 4)],
            "auc_difference_to_baseline": round(float(diff.mean()), 4), "auc_difference_ci95": [round(float(np.quantile(diff, 0.025)), 4), round(float(np.quantile(diff, 0.975)), 4)]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Our models and the CLIP LN-tuning baseline on the same videos: AUC, AP and macro-F1 at 0.5 with bootstrap intervals, and the paired AUC difference. Reads saved predictions only.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    report: dict = {}
    for scope, runs in OURS.items():
        base_path = config.paths.predictions_dir(f"clip_ln_tuning_on_{scope}") / "predictions.csv"
        if not base_path.exists():
            report[scope] = "baseline not scored yet"
            continue
        with open(base_path, encoding="utf-8", newline="") as f:
            base_rows = list(csv.DictReader(f))
        scored = {r["video_id"]: (int(r["label"]), float(r["p_fake"])) for r in base_rows if r["p_fake"] != ""}
        entry: dict = {"videos_in_list": len(base_rows), "videos_with_a_face_in_baseline": len(scored)}
        ours_all = {name: read_ours(config.paths.predictions_dir(folder) / "predictions.csv", DATASET_OF_SCOPE[scope]) for name, folder in runs.items()}
        common = sorted(set(scored).intersection(*[set(v) for v in ours_all.values()]))
        entry["videos_scored_by_everyone"] = len(common)
        if len(common) < 10:
            report[scope] = entry
            continue
        labels = np.array([scored[v][0] for v in common])
        base_scores = np.array([scored[v][1] for v in common])
        entry["real_videos"], entry["fake_videos"] = int((labels == 0).sum()), int((labels == 1).sum())
        entry["clip_ln_tuning"] = {**{k: round(v, 4) for k, v in metrics(labels, base_scores).items()}}
        for name, preds in ours_all.items():
            scores = np.array([preds[v] for v in common])
            entry[name] = {**{k: round(v, 4) for k, v in metrics(labels, scores).items()}, **bootstrap(labels, scores, base_scores, args.resamples, args.seed)}
        report[scope] = entry

    out_dir = config.paths.analysis_dir("clip_baseline_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
