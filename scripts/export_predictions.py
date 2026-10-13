from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import evaluate_model, test_records
from deepfake_detection.reference import load_reference_repo
from deepfake_detection.training.cv_splits import load_kfold_splits
from deepfake_detection.utils.progress import progress

CHECKPOINTS = ("best_f1", "swa")
# data.experiment of the old repo's configs -> (held-out dataset scored whole, scope name of the inference folder)
HELD_OUT = {
    "lodo_leave_polyglot": ("polyglotfake", "polyglotfake"), "lodo_leave_hidf": ("hidf", "hidf"), "lodo_leave_fakeavceleb": ("fakeavceleb", "fakeavceleb"),
    "lodo_leave_celebdfv2": ("celebdfv2", "celebdfv2"), "lodo_leave_ffpp": ("ffpp", "ffpp"),
    "ffpp_official_to_celebdfv2": ("celebdfv2", "celebdfv2_from_ffpp_four_types"), "ffpp5_official_to_celebdfv2": ("celebdfv2", "celebdfv2_from_ffpp_five_types"),
}


def records_of_run(reference, config, repro: dict, name: str, meta: dict) -> tuple[list, str]:
    """Test records and a scope name for a model folder: the exported runs come from reproduction.yaml,
    the retrained ones from the data section of their results.yaml."""
    if name in repro["models"]:
        experiment = repro["models"][name]
        return test_records(reference, config, experiment, repro["split"]), experiment["scope"]
    data = meta["data"]
    held_out = HELD_OUT.get(data.get("experiment"))
    if held_out:
        dataset, scope = held_out
        return test_records(reference, config, {"experiment": "leave_out", "held_out": dataset}, repro["split"]), scope
    if data.get("experiment") == "all_datasets_kfold":
        fold = int(data["fold"])
        return load_kfold_splits(reference, config, fold, int(data.get("folds", 5)), int(data.get("seed", 42)))[2], f"cv_fold{fold}_test"
    split = {"val_ratio": data.get("val_ratio", 0.2), "seed": data.get("seed", 42)}
    return test_records(reference, config, {"experiment": "combined"}, split), f"combined_test_seed{split['seed']}"


def group_metrics(labels: np.ndarray, probs: np.ndarray, alpha: np.ndarray | None) -> dict:
    row = {"videos": int(len(labels)), "real": int((labels == 0).sum()), "fake": int((labels == 1).sum()),
           "real_flagged_at_0.5": int(((labels == 0) & (probs >= 0.5)).sum()), "fake_missed_at_0.5": int(((labels == 1) & (probs < 0.5)).sum())}
    row["macro_f1"] = round(float(f1_score(labels, probs >= 0.5, average="macro")), 4)
    if len(set(labels.tolist())) == 2:
        row["auc"] = round(float(roc_auc_score(labels, probs)), 4)
        row["ap"] = round(float(average_precision_score(labels, probs)), 4)
    if alpha is not None and alpha.any():
        row["alpha_mean"] = round(float(alpha.mean()), 4)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the test split of finished models with their best_f1 and swa checkpoints and save one row per test video (id, dataset, subject, label, type, probability and gate weight) plus metrics overall and per dataset.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--models", nargs="*", default=None, help="model folder names; default: every folder under models/ with a results.yaml and no export yet")
    parser.add_argument("--force", action="store_true", help="export again even when predictions.csv exists")
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    models_root = config.paths.models_root
    names = args.models or sorted(p.parent.name for p in models_root.glob("*/results.yaml"))
    datasets_root = config.paths.datasets_root

    for name in names:
        model_dir = models_root / name
        meta = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))
        records, scope = records_of_run(reference, config, repro, name, meta)
        out_dir = config.paths.predictions_dir(f"{name}_on_{scope}")
        if (out_dir / "predictions.csv").exists() and not args.force:
            print(f"{name}: already exported")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        columns, summary = {}, {"model": name, "scope": scope, "videos": len(records)}
        bar = progress(len(CHECKPOINTS), 0, out_dir / "progress.log", f"{name} export")
        for checkpoint in CHECKPOINTS:
            if not (model_dir / f"{checkpoint}.pt").exists():
                bar.update(1)
                continue
            result = evaluate_model(reference, model_dir, records, repro["loader"], checkpoint=checkpoint)
            labels = np.array([int(r[1]) for r in records])
            assert np.array_equal(result["labels_arr"], labels), "prediction order differs from the record order"
            columns[f"p_fake_{checkpoint}"] = result["probs_arr"]
            columns[f"alpha_{checkpoint}"] = result["alpha_arr"] if result["alpha_arr"] is not None else np.full(len(labels), np.nan, dtype=np.float32)
            dataset_of = np.array([Path(r[0][0]).relative_to(datasets_root).parts[0] for r in records])
            alpha = result["alpha_arr"]
            summary[checkpoint] = {"overall": group_metrics(labels, result["probs_arr"], alpha),
                                   "per_dataset": {str(d): group_metrics(labels[dataset_of == d], result["probs_arr"][dataset_of == d], None if alpha is None else alpha[dataset_of == d]) for d in sorted(set(dataset_of))}}
            bar.update(1)
        bar.close()

        with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "dataset", "subject", "label", "manip_type", "audio_avail"] + list(columns))
            for i, (files, label, subject, manip, audio) in enumerate(records):
                relative = Path(files[0]).relative_to(datasets_root)
                writer.writerow([relative.as_posix(), relative.parts[0], subject, int(label), manip, int(bool(audio))] + [f"{columns[c][i]:.6f}" for c in columns])
        metrics_dir = config.paths.results_dir(f"evaluate_{name}")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "metrics.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
        print(f"{name}: {len(records)} videos exported to {out_dir.name}")


if __name__ == "__main__":
    main()
