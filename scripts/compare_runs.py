from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepfake_detection.config import load_config

DEFAULT_PAIRS = ["42=moe_combined_retrain:concat_combined_retrain", "43=moe_combined_seed43:concat_combined_seed43", "44=moe_combined_seed44:concat_combined_seed44"]
DEFAULT_CONTROLS = ["moe_no_gate_loss", "audio_only_pure", "video_only_pure"]
REFERENCE_RUN = "moe_combined_retrain"
METRICS = ("macro_f1", "auc", "ap")


def load_run(models_dir: Path, name: str) -> dict | None:
    path = models_dir / name / "alpha_primary.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    return {"macro_f1": float(f1_score(labels, probs >= 0.5, average="macro")), "auc": float(roc_auc_score(labels, probs)), "ap": float(average_precision_score(labels, probs))}


def describe(run: dict) -> dict:
    row = metrics(run["labels"], run["probs"])
    row["videos"] = int(len(run["labels"]))
    row["alpha_mean"] = float(run["alpha"].mean()) if run["alpha"].any() else None
    for code, kind in ((0, "real"), (1, "audio_video"), (2, "video_only")):
        mask = run["manip_type"] == code
        row[f"alpha_{kind}"] = float(run["alpha"][mask].mean()) if run["alpha"].any() and mask.any() else None
    return row


def bootstrap_difference(a: dict, b: dict, resamples: int, seed: int) -> dict:
    """Difference a minus b on the same test videos, resampling videos with replacement."""
    rng = np.random.default_rng(seed)
    n = len(a["labels"])
    draws = {m: [] for m in METRICS}
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        y = a["labels"][index]
        if y.min() == y.max():
            continue
        left, right = metrics(y, a["probs"][index]), metrics(y, b["probs"][index])
        for m in METRICS:
            draws[m].append(left[m] - right[m])
    observed = {m: metrics(a["labels"], a["probs"])[m] - metrics(b["labels"], b["probs"])[m] for m in METRICS}
    return {m: {"difference": round(observed[m], 4), "ci95": [round(float(np.percentile(draws[m], 2.5)), 4), round(float(np.percentile(draws[m], 97.5)), 4)]} for m in METRICS}


def cross_validation(models_dir: Path, gated_prefix: str, static_prefix: str, folds: int, resamples: int, seed: int) -> dict:
    """Out-of-fold results: per-fold metrics, the predictions of all folds pooled (every video is tested once),
    and the gated minus static difference per fold and on the pooled videos."""
    pairs = []
    for fold in range(folds):
        gated, static = load_run(models_dir, f"{gated_prefix}{fold}"), load_run(models_dir, f"{static_prefix}{fold}")
        if gated is not None and static is not None and np.array_equal(gated["labels"], static["labels"]):
            pairs.append((fold, gated, static))
    if not pairs:
        return {}
    result: dict = {"folds_done": [fold for fold, _, _ in pairs], "per_fold": {}, "mean_sd": {}, "pooled": {}}
    per_model = {"gated": {m: [] for m in METRICS}, "static": {m: [] for m in METRICS}}
    for fold, gated, static in pairs:
        row = {}
        for label, run in (("gated", gated), ("static", static)):
            values = metrics(run["labels"], run["probs"])
            row[label] = {m: round(v, 4) for m, v in values.items()}
            for m in METRICS:
                per_model[label][m].append(values[m])
        result["per_fold"][fold] = row
    for m in METRICS:
        g, s = np.array(per_model["gated"][m]), np.array(per_model["static"][m])
        entry = {"gated_mean": round(float(g.mean()), 4), "static_mean": round(float(s.mean()), 4)}
        if len(g) > 1:
            entry.update({"gated_sd": round(float(g.std(ddof=1)), 4), "static_sd": round(float(s.std(ddof=1)), 4)})
        if len(g) > 2:
            entry["paired_t_p_value"] = round(float(stats.ttest_rel(g, s).pvalue), 4)
            entry["wilcoxon_p_value"] = round(float(stats.wilcoxon(g, s).pvalue), 4)
        result["mean_sd"][m] = entry
    pooled = {}
    for label, position in (("gated", 1), ("static", 2)):
        pooled[label] = {"labels": np.concatenate([item[position]["labels"] for item in pairs]), "probs": np.concatenate([item[position]["probs"] for item in pairs])}
    result["pooled"] = {"videos": int(len(pooled["gated"]["labels"])),
                        "gated": {m: round(v, 4) for m, v in metrics(pooled["gated"]["labels"], pooled["gated"]["probs"]).items()},
                        "static": {m: round(v, 4) for m, v in metrics(pooled["static"]["labels"], pooled["static"]["probs"]).items()},
                        "gated_minus_static": bootstrap_difference(pooled["gated"], pooled["static"], resamples, seed)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the retrained runs: gated model against static concatenation per seed with a paired bootstrap on the same test videos, spread over seeds with a paired t-test, and the controls against the reference run.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS, help="seed=gated_run:static_run")
    parser.add_argument("--controls", nargs="*", default=DEFAULT_CONTROLS)
    parser.add_argument("--reference", default=REFERENCE_RUN)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="training_comparison")
    parser.add_argument("--cv-gated", default="moe_cv_fold", help="prefix of the gated cross-validation runs, the fold number follows")
    parser.add_argument("--cv-static", default="concat_cv_fold")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    models_dir = config.paths.models_root
    report: dict = {"runs": {}, "per_seed": {}, "across_seeds": {}, "controls_against_reference": {}}

    names = [n for pair in args.pairs for n in pair.split("=")[1].split(":")] + args.controls
    loaded = {n: load_run(models_dir, n) for n in dict.fromkeys(names)}
    for name, run in loaded.items():
        if run is not None:
            report["runs"][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in describe(run).items()}

    gated_scores, static_scores = {m: [] for m in METRICS}, {m: [] for m in METRICS}
    for pair in args.pairs:
        seed, runs = pair.split("=")
        gated_name, static_name = runs.split(":")
        gated, static = loaded.get(gated_name), loaded.get(static_name)
        if gated is None or static is None:
            continue
        if not np.array_equal(gated["labels"], static["labels"]):
            report["per_seed"][seed] = "test videos differ between the two runs, no paired comparison"
            continue
        report["per_seed"][seed] = {"gated_minus_static": bootstrap_difference(gated, static, args.resamples, args.seed)}
        for m in METRICS:
            gated_scores[m].append(metrics(gated["labels"], gated["probs"])[m])
            static_scores[m].append(metrics(static["labels"], static["probs"])[m])

    seeds_used = len(gated_scores["auc"])
    if seeds_used:
        for m in METRICS:
            g, s = np.array(gated_scores[m]), np.array(static_scores[m])
            entry = {"seeds": seeds_used, "gated_mean": round(float(g.mean()), 4), "gated_sd": round(float(g.std(ddof=1)), 4) if seeds_used > 1 else None,
                     "static_mean": round(float(s.mean()), 4), "static_sd": round(float(s.std(ddof=1)), 4) if seeds_used > 1 else None}
            if seeds_used > 2:
                test = stats.ttest_rel(g, s)
                entry["paired_t_p_value"] = round(float(test.pvalue), 4)
            report["across_seeds"][m] = entry

    reference = loaded.get(args.reference)
    for name in args.controls:
        run = loaded.get(name)
        if run is None or reference is None:
            continue
        entry = {"metrics": report["runs"][name]}
        if np.array_equal(run["labels"], reference["labels"]):
            entry["minus_reference"] = bootstrap_difference(run, reference, args.resamples, args.seed)
        report["controls_against_reference"][name] = entry

    report["cross_validation"] = cross_validation(models_dir, args.cv_gated, args.cv_static, args.folds, args.resamples, args.seed)

    out_dir = config.paths.results_dir(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print("run | videos | macro-F1 | AUC | AP | mean alpha | alpha audio-video | alpha video-only | alpha real")
    for name, row in report["runs"].items():
        shown = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"{name} | {row['videos']} | {row['macro_f1']:.4f} | {row['auc']:.4f} | {row['ap']:.4f} | {shown(row['alpha_mean'])} | {shown(row['alpha_audio_video'])} | {shown(row['alpha_video_only'])} | {shown(row['alpha_real'])}")
    print(yaml.safe_dump({k: report[k] for k in ("per_seed", "across_seeds", "controls_against_reference")}, sort_keys=False))


if __name__ == "__main__":
    main()
