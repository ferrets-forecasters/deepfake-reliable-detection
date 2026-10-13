from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from deepfake_detection.config import load_config

GATED, STATIC = "moe_combined_retrain", "concat_combined_retrain"
SEED_RUNS = {"42": (GATED, STATIC), "43": ("moe_combined_seed43", "concat_combined_seed43"), "44": ("moe_combined_seed44", "concat_combined_seed44")}
ABLATION = [("Static concatenation", STATIC), ("Gated, with gate loss", GATED), ("Gated, without gate loss", "moe_no_gate_loss"), ("Audio only", "audio_only_pure"), ("Video only", "video_only_pure"),
            ("Audio-visual branch alone", "audio_visual_branch_alone"), ("Visual branch alone", "visual_branch_alone")]
LEAVE_OUT = [("polyglotfake", "moe_leave_polyglotfake"), ("hidf", "moe_leave_hidf"), ("fakeavceleb", "moe_leave_fakeavceleb"), ("celebdfv2", "moe_leave_celebdfv2"), ("ffpp", "moe_leave_ffpp")]
FOLDS = 5
DATASETS = ["polyglotfake", "hidf", "fakeavceleb", "celebdfv2", "ffpp"]
DATASET_LABELS = {"polyglotfake": "PolyglotFake", "hidf": "HiDF", "fakeavceleb": "FakeAVCeleb", "celebdfv2": "CelebDFv2", "ffpp": "FF++"}
ALPHA_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def load_predictions(config, name: str, checkpoint: str = "best_f1") -> dict | None:
    matches = sorted(config.paths.inferences_root.glob(f"{name}_on_*/predictions.csv"))
    if not matches:
        return None
    with open(matches[0], encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    column = f"p_fake_{checkpoint}"
    if not rows or column not in rows[0]:
        return None
    alpha = np.array([float(r[f"alpha_{checkpoint}"]) for r in rows])
    return {"video": np.array([r["video"] for r in rows]), "dataset": np.array([r["dataset"] for r in rows]), "label": np.array([int(r["label"]) for r in rows]),
            "kind": np.array([r["manip_type"] for r in rows]), "probs": np.array([float(r[column]) for r in rows]), "alpha": alpha}


def scores(label: np.ndarray, probs: np.ndarray) -> dict:
    return {"macro_f1": float(f1_score(label, probs >= 0.5, average="macro")), "auc": float(roc_auc_score(label, probs)), "ap": float(average_precision_score(label, probs))}


def fmt(value, digits: int = 4) -> str:
    return "-" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def alpha_means(run: dict) -> dict:
    out = {"mean": None, "audio_video": None, "video_only": None, "real": None}
    if np.isnan(run["alpha"]).all():
        return out
    out["mean"] = float(np.nanmean(run["alpha"]))
    for kind in ("audio_video", "video_only", "real"):
        mask = run["kind"] == kind
        out[kind] = float(np.nanmean(run["alpha"][mask])) if mask.any() else None
    return out


def bootstrap(a: dict, b: dict, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a["label"])
    draws = {m: [] for m in ("macro_f1", "auc", "ap")}
    for _ in range(resamples):
        index = rng.integers(0, n, n)
        if a["label"][index].min() == a["label"][index].max():
            continue
        left, right = scores(a["label"][index], a["probs"][index]), scores(a["label"][index], b["probs"][index])
        for m in draws:
            draws[m].append(left[m] - right[m])
    observed = {m: scores(a["label"], a["probs"])[m] - scores(a["label"], b["probs"])[m] for m in draws}
    return {m: (observed[m], float(np.percentile(draws[m], 2.5)), float(np.percentile(draws[m], 97.5))) for m in draws}


def table(title: str, header: list[str], rows: list[list[str]]) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Every table the paper needs from the finished training runs, in fixed columns: main comparison, ablation, seeds, cross validation, per dataset, gate weight. Runs that are not finished show as '-'. Exports the per-video predictions first unless --skip-export.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(args.config)
    if not args.skip_export:
        subprocess.run([sys.executable, str(Path(__file__).with_name("export_predictions.py"))], check=True)

    cache: dict[str, dict | None] = {}
    get = lambda name: cache.setdefault(name, load_predictions(config, name))
    sections, report = [], {}

    rows = []
    for label, name in ABLATION:
        run = get(name)
        if run is None:
            rows.append([label, "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        s, a = scores(run["label"], run["probs"]), alpha_means(run)
        real, fake = run["label"] == 0, run["label"] == 1
        rows.append([label, fmt(s["macro_f1"]), fmt(s["auc"]), fmt(s["ap"]), fmt(a["mean"], 3), fmt(a["audio_video"], 3), fmt(a["video_only"], 3), fmt(a["real"], 3),
                     str(int(((run["probs"] >= 0.5) & real).sum())), str(int(((run["probs"] < 0.5) & fake).sum()))])
        report.setdefault("main", {})[name] = {**s, **{f"alpha_{k}": v for k, v in a.items()}}
    sections.append(table("Main comparison and ablation (seed 42 split)", ["Model", "Macro-F1", "AUC", "AP", "Mean α", "α audio-video fakes", "α video-only fakes", "α real", "Real flagged at 0.5", "Fakes missed at 0.5"], rows))

    rows, gated_all, static_all = [], {m: [] for m in ("macro_f1", "auc", "ap")}, {m: [] for m in ("macro_f1", "auc", "ap")}
    for seed, (gated_name, static_name) in SEED_RUNS.items():
        g, s = get(gated_name), get(static_name)
        gs = scores(g["label"], g["probs"]) if g is not None else None
        ss = scores(s["label"], s["probs"]) if s is not None else None
        rows.append([seed, fmt(gs and gs["macro_f1"]), fmt(ss and ss["macro_f1"]), fmt(gs and gs["auc"]), fmt(ss and ss["auc"]), fmt(gs and gs["ap"]), fmt(ss and ss["ap"])])
        if gs and ss:
            for m in gated_all:
                gated_all[m].append(gs[m])
                static_all[m].append(ss[m])
    n = len(gated_all["auc"])
    summary = ["mean ± sd"]
    p_values = ["paired t-test p"]
    for m in ("macro_f1", "auc", "ap"):
        for data in (gated_all, static_all):
            summary.append("-" if n < 2 else f"{np.mean(data[m]):.4f} ± {np.std(data[m], ddof=1):.4f}")
        p_values += ["-" if n < 3 else f"{stats.ttest_rel(gated_all[m], static_all[m]).pvalue:.4f}", ""]
    rows += [summary, p_values]
    sections.append(table(f"Seeds ({n} finished; each seed has its own subject-level split)", ["Seed", "Macro-F1 gated", "Macro-F1 static", "AUC gated", "AUC static", "AP gated", "AP static"], rows))

    rows = []
    for seed, (gated_name, static_name) in SEED_RUNS.items():
        g, s = get(gated_name), get(static_name)
        if g is None or s is None or not np.array_equal(g["video"], s["video"]):
            rows.append([seed, "-", "-", "-", "-"])
            continue
        right_g, right_s = (g["probs"] >= 0.5) == g["label"].astype(bool), (s["probs"] >= 0.5) == s["label"].astype(bool)
        only_g, only_s = int((right_g & ~right_s).sum()), int((~right_g & right_s).sum())
        p = stats.binomtest(only_g, only_g + only_s, 0.5).pvalue if only_g + only_s else 1.0
        rows.append([seed, str(len(g["label"])), str(only_g), str(only_s), f"{p:.4f}"])
    sections.append(table("McNemar test at 0.5, gated against static on the same test videos", ["Seed", "Videos", "Only gated right", "Only static right", "Exact p"], rows))

    rows = []
    for label, (may_name, rerun_name) in (("Gated", ("moe_combined", GATED)), ("Static", ("concat_combined", STATIC))):
        may, rerun = get(may_name), get(rerun_name)
        if may is None or rerun is None or not np.array_equal(may["video"], rerun["video"]):
            rows.append([label, "-", "-", "-", "-", "-"])
            continue
        a, b = scores(may["label"], may["probs"]), scores(rerun["label"], rerun["probs"])
        diff = bootstrap(rerun, may, args.resamples, args.seed)
        rows.append([label, f"{a['macro_f1']:.4f} to {b['macro_f1']:.4f}", f"{a['auc']:.4f} to {b['auc']:.4f}", f"{a['ap']:.4f} to {b['ap']:.4f}",
                     f"{diff['macro_f1'][0]:+.4f} ({diff['macro_f1'][1]:+.4f} to {diff['macro_f1'][2]:+.4f})", f"{diff['auc'][0]:+.4f} ({diff['auc'][1]:+.4f} to {diff['auc'][2]:+.4f})"])
    sections.append(table("May run against re-run, same split and test videos (re-run minus May, 95% interval)", ["Model", "Macro-F1, May to re-run", "AUC, May to re-run", "AP, May to re-run", "Macro-F1 difference", "AUC difference"], rows))

    rows, pooled = [], {"gated": [], "static": []}
    per_fold = {"gated": {m: [] for m in ("macro_f1", "auc", "ap")}, "static": {m: [] for m in ("macro_f1", "auc", "ap")}}
    for fold in range(FOLDS):
        g, s = get(f"moe_cv_fold{fold}"), get(f"concat_cv_fold{fold}")
        gs = scores(g["label"], g["probs"]) if g is not None else None
        ss = scores(s["label"], s["probs"]) if s is not None else None
        rows.append([str(fold), fmt(gs and gs["macro_f1"]), fmt(ss and ss["macro_f1"]), fmt(gs and gs["auc"]), fmt(ss and ss["auc"]), fmt(gs and gs["ap"]), fmt(ss and ss["ap"])])
        if g is not None and s is not None and np.array_equal(g["video"], s["video"]):
            pooled["gated"].append(g)
            pooled["static"].append(s)
            for m in per_fold["gated"]:
                per_fold["gated"][m].append(gs[m])
                per_fold["static"][m].append(ss[m])
    folds_done = len(pooled["gated"])
    mean_row, p_row = ["mean ± sd"], ["paired t-test p, Wilcoxon p"]
    for m in ("macro_f1", "auc", "ap"):
        for side in ("gated", "static"):
            v = per_fold[side][m]
            mean_row.append("-" if folds_done < 2 else f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}")
        if folds_done >= 3:
            p_row += [f"{stats.ttest_rel(per_fold['gated'][m], per_fold['static'][m]).pvalue:.4f}, {stats.wilcoxon(per_fold['gated'][m], per_fold['static'][m]).pvalue:.4f}", ""]
        else:
            p_row += ["-", ""]
    rows += [mean_row, p_row]
    if folds_done:
        merged = {side: {k: np.concatenate([r[k] for r in pooled[side]]) for k in ("label", "probs")} for side in pooled}
        both = bootstrap(merged["gated"], merged["static"], args.resamples, args.seed)
        pooled_scores = {side: scores(merged[side]["label"], merged[side]["probs"]) for side in merged}
        rows.append(["pooled out-of-fold", fmt(pooled_scores["gated"]["macro_f1"]), fmt(pooled_scores["static"]["macro_f1"]), fmt(pooled_scores["gated"]["auc"]), fmt(pooled_scores["static"]["auc"]), fmt(pooled_scores["gated"]["ap"]), fmt(pooled_scores["static"]["ap"])])
        rows.append(["gated minus static, 95% interval"] + [f"{both[m][0]:+.4f} ({both[m][1]:+.4f} to {both[m][2]:+.4f})" if i % 2 == 0 else "" for m in ("macro_f1", "auc", "ap") for i in range(2)])
    sections.append(table(f"Five-fold cross validation ({folds_done} of {FOLDS} folds finished)", ["Fold", "Macro-F1 gated", "Macro-F1 static", "AUC gated", "AUC static", "AP gated", "AP static"], rows))

    rows = []
    g, s = get(GATED), get(STATIC)
    for dataset in DATASETS:
        row = [DATASET_LABELS[dataset]]
        for run in (g, s):
            if run is None or not (run["dataset"] == dataset).any():
                row += ["-", "-"]
                continue
            mask = run["dataset"] == dataset
            if len(set(run["label"][mask].tolist())) < 2:
                row += ["-", "-"]
                continue
            v = scores(run["label"][mask], run["probs"][mask])
            row += [fmt(v["macro_f1"]), fmt(v["auc"])]
        a = (np.nanmean(g["alpha"][g["dataset"] == dataset]) if g is not None and (g["dataset"] == dataset).any() else None)
        rows.append(row + [fmt(a, 3)])
    sections.append(table("Per dataset (seed 42 split)", ["Dataset", "Macro-F1 gated", "AUC gated", "Macro-F1 static", "AUC static", "Mean α gated"], rows))

    rows = []
    if g is not None:
        for low, high in zip(ALPHA_EDGES[:-1], ALPHA_EDGES[1:]):
            mask = (g["alpha"] >= low) & ((g["alpha"] < high) if high < 1.0 else (g["alpha"] <= high))
            if mask.any():
                wrong = (g["probs"][mask] >= 0.5) != g["label"][mask].astype(bool)
                rows.append([f"{low:.1f} to {high:.1f}", str(int(mask.sum())), fmt(float(g["label"][mask].mean()), 3), fmt(float(wrong.mean()))])
    sections.append(table("Error rate by gate weight (gated, seed 42 split)", ["α", "Videos", "Fake share", "Error rate at 0.5"], rows))

    rows = []
    n_ref = get(GATED)
    for label, name in (("With gate loss", GATED), ("Without gate loss", "moe_no_gate_loss")):
        run = get(name)
        for dataset in DATASETS:
            if run is None or not (run["dataset"] == dataset).any():
                continue
            mask = run["dataset"] == dataset
            row = [label, DATASET_LABELS[dataset]]
            for kind in ("audio_video", "video_only", "real"):
                m = mask & (run["kind"] == kind)
                row.append(fmt(float(np.nanmean(run["alpha"][m])), 3) if m.any() else "-")
            rows.append(row)
    sections.append(table("Gate weight per dataset and manipulation type", ["Run", "Dataset", "α audio-video", "α video-only", "α real"], rows))

    if n_ref is not None and get("moe_no_gate_loss") is not None:
        diff = bootstrap(get(GATED), get("moe_no_gate_loss"), args.resamples, args.seed)
        sections.append(table("Gate loss, difference with minus without (95% interval)", ["Macro-F1", "AUC", "AP"], [[f"{diff[m][0]:+.4f} ({diff[m][1]:+.4f} to {diff[m][2]:+.4f})" for m in ("macro_f1", "auc", "ap")]]))
    if g is not None and s is not None and np.array_equal(g["video"], s["video"]):
        diff = bootstrap(g, s, args.resamples, args.seed)
        sections.append(table("Gated minus static, seed 42 split (95% interval)", ["Macro-F1", "AUC", "AP"], [[f"{diff[m][0]:+.4f} ({diff[m][1]:+.4f} to {diff[m][2]:+.4f})" for m in ("macro_f1", "auc", "ap")]]))

    rows = []
    for dataset, may_name in LEAVE_OUT:
        row = [DATASET_LABELS[dataset]]
        for name in (may_name, f"{may_name}_retrain"):
            path = config.paths.checkpoints_dir(name) / "results.yaml"
            result = yaml.safe_load(path.read_text(encoding="utf-8"))["test_results"]["primary_best_f1"] if path.exists() else None
            row += [fmt(result and result["f1"]), fmt(result and result["auc"]), fmt(result and result["ap"]), fmt(result and result["alpha_mean"], 3)]
        rows.append(row)
    sections.append(table("Leave-one-out, May run against re-run (whole held-out dataset)", ["Held out", "F1 May", "AUC May", "AP May", "Mean α May", "F1 re-run", "AUC re-run", "AP re-run", "Mean α re-run"], rows))

    out_dir = config.paths.results_dir("training_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    curves =[(label, get(name)) for label, name in ABLATION if get(name) is not None]
    if curves:
        fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(9, 4))
        for index, (label, run) in enumerate(curves):
            style = ["-", "--", ":", "-.", (0, (1, 3)), (0, (5, 2)), (0, (3, 1, 1, 1))][index % 7]
            colour = "#c8463c" if "with gate loss" in label else "#222222"
            fpr, tpr, _ = roc_curve(run["label"], run["probs"])
            precision, recall, _ = precision_recall_curve(run["label"], run["probs"])
            roc_ax.plot(fpr, tpr, linestyle=style, color=colour, label=label)
            pr_ax.plot(recall, precision, linestyle=style, color=colour, label=label)
        roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
        pr_ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
        for ax in (roc_ax, pr_ax):
            ax.spines[["top", "right"]].set_visible(False)
        roc_ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "roc_pr_ablation.png", dpi=200)
        plt.close(fig)
    (out_dir / "report.md").write_text("# Training report\n\n" + "\n".join(sections), encoding="utf-8")
    (out_dir / "main.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print("\n".join(sections))


if __name__ == "__main__":
    main()
