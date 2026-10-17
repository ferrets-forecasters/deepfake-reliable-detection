"""Gated fusion against static concatenation on the same videos: per dataset and per manipulation type, from the saved predictions.
Five-fold cross validation gives every video one gated and one static score, so each comparison is paired. Writes
results/analyses/gate_vs_static/: per_dataset.csv (metrics with paired bootstrap intervals and the McNemar test), per_seed.csv (the three
seeds), examples.csv (the videos where one model is right with a clear margin and the other wrong) and summary.yaml."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepfake_detection.config import load_config

OUT = "gate_vs_static"
SEEDS = {42: ("moe_combined_retrain_on_combined_test_seed42", "concat_combined_retrain_on_combined_test_seed42"),
         43: ("moe_combined_seed43_on_combined_test_seed43", "concat_combined_seed43_on_combined_test_seed43"),
         44: ("moe_combined_seed44_on_combined_test_seed44", "concat_combined_seed44_on_combined_test_seed44")}
COLUMN = "p_fake_best_f1"


def load_folds(root: Path) -> pd.DataFrame:
    gated = pd.concat([pd.read_csv(root / f"moe_cv_fold{i}_on_cv_fold{i}_test" / "predictions.csv") for i in range(5)], ignore_index=True)
    static = pd.concat([pd.read_csv(root / f"concat_cv_fold{i}_on_cv_fold{i}_test" / "predictions.csv") for i in range(5)], ignore_index=True)
    assert (gated.video.to_numpy() == static.video.to_numpy()).all(), "fold predictions are not in the same video order"
    gated["p_static"] = static[COLUMN].to_numpy()
    return gated.rename(columns={COLUMN: "p_gate", "alpha_best_f1": "alpha"})


def metrics(label: np.ndarray, p: np.ndarray) -> dict:
    row = {"f1": f1_score(label, p >= 0.5, average="macro")}
    if len(set(label.tolist())) == 2:
        row["auc"], row["ap"] = roc_auc_score(label, p), average_precision_score(label, p)
    return row


def paired_bootstrap(label: np.ndarray, pg: np.ndarray, ps: np.ndarray, rng: np.random.Generator, resamples: int) -> dict:
    """95% interval of gated minus static for each metric, resampling videos."""
    differences = {"f1": [], "auc": [], "ap": []}
    n = len(label)
    for _ in range(resamples):
        pick = rng.integers(0, n, n)
        if len(set(label[pick].tolist())) < 2:
            continue
        g, s = metrics(label[pick], pg[pick]), metrics(label[pick], ps[pick])
        for key in differences:
            differences[key].append(g[key] - s[key])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in differences.items()}


def group_row(name: str, frame: pd.DataFrame, rng: np.random.Generator, resamples: int) -> dict:
    label, pg, ps = frame.label.to_numpy(), frame.p_gate.to_numpy(), frame.p_static.to_numpy()
    g, s = metrics(label, pg), metrics(label, ps)
    right_g, right_s = (pg >= 0.5) == (label == 1), (ps >= 0.5) == (label == 1)
    only_gate, only_static = int((right_g & ~right_s).sum()), int((~right_g & right_s).sum())
    p_value = binomtest(only_gate, only_gate + only_static, 0.5).pvalue if only_gate + only_static else 1.0
    row = {"group": name, "videos": len(frame), "real": int((label == 0).sum()), "fake": int((label == 1).sum()),
           "f1_gate": g["f1"], "f1_static": s["f1"], "gate_right_only": only_gate, "static_right_only": only_static, "mcnemar_p": p_value}
    if "auc" in g:
        row.update({"auc_gate": g["auc"], "auc_static": s["auc"], "ap_gate": g["ap"], "ap_static": s["ap"]})
        interval = paired_bootstrap(label, pg, ps, rng, resamples)
        for key, (low, high) in interval.items():
            row[f"{key}_diff"] = {"f1": g["f1"] - s["f1"], "auc": g["auc"] - s["auc"], "ap": g["ap"] - s["ap"]}[key]
            row[f"{key}_low"], row[f"{key}_high"] = low, high
    return row


def triage(frame: pd.DataFrame, threshold: float = 0.4) -> dict:
    """Which videos to send to a human: those with alpha at or above the threshold, against the same number of least confident videos
    of each model. Reports the share of that model's errors the review set holds and how well each signal ranks the errors (AUC)."""
    label = frame.label.to_numpy()
    wrong_g, wrong_s = (frame.p_gate >= 0.5).to_numpy() != (label == 1), (frame.p_static >= 0.5).to_numpy() != (label == 1)
    review = (frame.alpha >= threshold).to_numpy()
    size = int(review.sum())
    uncertain_g = np.argsort(np.abs(frame.p_gate.to_numpy() - 0.5))[:size]
    uncertain_s = np.argsort(np.abs(frame.p_static.to_numpy() - 0.5))[:size]
    return {"videos": len(frame), "review_share": size / len(frame), "gate_errors": int(wrong_g.sum()), "static_errors": int(wrong_s.sum()),
            "gate_errors_in_alpha_set": float(wrong_g[review].mean()) if size else np.nan,
            "gate_errors_outside_alpha_set": float(wrong_g[~review].mean()),
            "share_of_gate_errors_in_alpha_set": float(wrong_g[review].sum() / max(1, wrong_g.sum())),
            "share_of_gate_errors_in_least_confident_gate": float(wrong_g[uncertain_g].sum() / max(1, wrong_g.sum())),
            "share_of_static_errors_in_least_confident_static": float(wrong_s[uncertain_s].sum() / max(1, wrong_s.sum())),
            "error_auc_alpha": float(roc_auc_score(wrong_g, frame.alpha)),
            "error_auc_gate_confidence": float(roc_auc_score(wrong_g, -np.abs(frame.p_gate - 0.5))),
            "error_auc_static_confidence": float(roc_auc_score(wrong_s, -np.abs(frame.p_static - 0.5)))}


def examples(frame: pd.DataFrame, per_group: int) -> pd.DataFrame:
    """Videos where one model is right at 0.5 and the other wrong, ranked by how much more probability mass the right one puts on the truth."""
    label = frame.label.to_numpy()
    truth_g = np.where(label == 1, frame.p_gate, 1 - frame.p_gate)
    truth_s = np.where(label == 1, frame.p_static, 1 - frame.p_static)
    frame = frame.assign(gate_right=truth_g >= 0.5, static_right=truth_s >= 0.5, margin=truth_g - truth_s)
    frame["kind"] = np.where(frame.label == 1, frame.manip_type + " fake", "real")
    gate_wins = frame[frame.gate_right & ~frame.static_right].sort_values("margin", ascending=False)
    static_wins = frame[~frame.gate_right & frame.static_right].sort_values("margin")
    picks = []
    for winner, table in (("gated", gate_wins), ("static", static_wins)):
        for (dataset, kind), part in table.groupby(["dataset", "kind"]):
            picks.append(part.head(per_group).assign(winner=winner))
    return pd.concat(picks).loc[:, ["winner", "dataset", "kind", "video", "label", "p_gate", "p_static", "alpha", "margin"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--per-group", type=int, default=3)
    args = parser.parse_args()
    config = load_config(args.config)
    root = config.paths.inferences_root
    out_dir = config.paths.analysis_dir(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    folds = load_folds(root)
    rows = [group_row("all videos", folds, rng, args.resamples)]
    for dataset, part in folds.groupby("dataset"):
        rows.append(group_row(dataset, part, rng, args.resamples))
    for manip, part in folds[folds.label == 1].groupby("manip_type"):
        reals = folds[(folds.label == 0)]
        rows.append(group_row(f"fakes of type {manip} against all reals", pd.concat([part, reals]), rng, args.resamples))
    pd.DataFrame(rows).round(4).to_csv(out_dir / "per_dataset.csv", index=False)

    seed_rows = []
    for seed, (gated_name, static_name) in SEEDS.items():
        gated, static = pd.read_csv(root / gated_name / "predictions.csv"), pd.read_csv(root / static_name / "predictions.csv")
        assert (gated.video.to_numpy() == static.video.to_numpy()).all()
        frame = gated.rename(columns={COLUMN: "p_gate", "alpha_best_f1": "alpha"}).assign(p_static=static[COLUMN].to_numpy())
        for dataset, part in [("all videos", frame)] + list(frame.groupby("dataset")):
            row = group_row(dataset, part, rng, 200)
            seed_rows.append({"seed": seed, **row})
    pd.DataFrame(seed_rows).round(4).to_csv(out_dir / "per_seed.csv", index=False)
    triage_rows = []
    for seed, (gated_name, static_name) in SEEDS.items():
        gated, static = pd.read_csv(root / gated_name / "predictions.csv"), pd.read_csv(root / static_name / "predictions.csv")
        frame = gated.rename(columns={COLUMN: "p_gate", "alpha_best_f1": "alpha"}).assign(p_static=static[COLUMN].to_numpy())
        triage_rows.append({"seed": seed, **triage(frame)})
    pd.DataFrame(triage_rows).round(4).to_csv(out_dir / "triage.csv", index=False)

    table = examples(folds, args.per_group)
    table.round(4).to_csv(out_dir / "examples.csv", index=False)
    # the same on the seed-42 test split, whose HiDF videos are on disk, so a case can be shown as a clip
    gated42, static42 = pd.read_csv(root / SEEDS[42][0] / "predictions.csv"), pd.read_csv(root / SEEDS[42][1] / "predictions.csv")
    frame42 = gated42.rename(columns={COLUMN: "p_gate", "alpha_best_f1": "alpha"}).assign(p_static=static42[COLUMN].to_numpy())
    hidf_raw = {p.stem: p for p in (config.paths.datasets_root / "hidf" / "raw" / "test_split").rglob("*.mp4")}
    table42 = examples(frame42, args.per_group)
    table42["raw_video"] = [str(hidf_raw.get(Path(v).stem.replace("_seg000", ""), "")) if d == "hidf" else "" for v, d in zip(table42.video, table42.dataset)]
    table42.round(4).to_csv(out_dir / "examples_seed42.csv", index=False)
    summary = {"folds_videos": len(folds), "bootstrap_resamples": args.resamples,
               "videos_gate_right_only": int(((folds.p_gate >= 0.5) == (folds.label == 1))[(folds.p_static >= 0.5) != (folds.label == 1)].sum()),
               "videos_static_right_only": int(((folds.p_static >= 0.5) == (folds.label == 1))[(folds.p_gate >= 0.5) != (folds.label == 1)].sum())}
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string())
    print(summary)


if __name__ == "__main__":
    main()
