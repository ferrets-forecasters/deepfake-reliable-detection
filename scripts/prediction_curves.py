from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from deepfake_detection.config import load_config

COMBINED = {"moe_combined": "Gated MoE", "concat_combined": "Static concatenation"}
HELD_OUT = {"moe_leave_polyglotfake": "PolyglotFake", "moe_leave_hidf": "HiDF", "moe_leave_fakeavceleb": "FakeAVCeleb", "moe_leave_celebdfv2": "CelebDFv2", "moe_leave_ffpp": "FF++"}
SCOPES = {"moe_combined": "combined_test", "concat_combined": "combined_test", "moe_leave_polyglotfake": "polyglotfake", "moe_leave_hidf": "hidf",
          "moe_leave_fakeavceleb": "fakeavceleb", "moe_leave_celebdfv2": "celebdfv2", "moe_leave_ffpp": "ffpp"}
INK, ACCENT = "#222222", "#c8463c"


def load(config, model: str) -> dict:
    path = config.paths.predictions_dir(f"{model}_on_{SCOPES[model]}") / "predictions.npz"
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def curve_figure(items: dict[str, dict], path: Path, title_suffix: str) -> None:
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(9, 4))
    for index, (name, d) in enumerate(items.items()):
        color = ACCENT if index == 0 else INK
        style = "-" if index == 0 else "--" if index == 1 else ":"
        fpr, tpr, _ = roc_curve(d["labels"], d["probs"])
        precision, recall, _ = precision_recall_curve(d["labels"], d["probs"])
        roc_ax.plot(fpr, tpr, style, color=color, label=f"{name} (AUC {roc_auc_score(d['labels'], d['probs']):.3f})")
        pr_ax.plot(recall, precision, style, color=color, label=f"{name} (AP {average_precision_score(d['labels'], d['probs']):.3f})")
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC{title_suffix}")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall{title_suffix}")
    for ax in (roc_ax, pr_ax):
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def alpha_bins(d: dict, edges: list[float]) -> list[dict]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (d["alpha"] >= low) & ((d["alpha"] < high) if high < 1.0 else (d["alpha"] <= high))
        if not mask.any():
            continue
        wrong = (d["probs"][mask] >= 0.5) != d["labels"][mask].astype(bool)
        rows.append({"alpha_from": low, "alpha_to": high, "videos": int(mask.sum()), "fake_share": round(float(d["labels"][mask].mean()), 3),
                     "error_rate": round(float(wrong.mean()), 4)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="ROC and precision-recall curves, confusion counts and error rate by gate weight from the saved predictions of the exported models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.results_dir("prediction_curves")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    combined = {label: load(config, name) for name, label in COMBINED.items()}
    curve_figure(combined, out_dir / "roc_pr_combined_test.png", ", combined test split")
    held_out = {label: load(config, name) for name, label in HELD_OUT.items()}
    curve_figure(held_out, out_dir / "roc_pr_leave_one_out.png", ", held-out dataset")

    for name, label in {**COMBINED, **HELD_OUT}.items():
        d = load(config, name)
        tn, fp, fn, tp = confusion_matrix(d["labels"], d["probs"] >= 0.5, labels=[0, 1]).ravel()
        report[name] = {"scope": SCOPES[name], "videos": int(len(d["labels"])), "confusion_at_0.5": {"real_correct": int(tn), "real_flagged": int(fp), "fake_missed": int(fn), "fake_caught": int(tp)},
                        "probabilities_exactly_0.5": int((d["probs"] == 0.5).sum())}

    edges = [round(i / args.bins, 3) for i in range(args.bins + 1)]
    gated = load(config, "moe_combined")
    report["moe_combined"]["error_by_alpha"] = alpha_bins(gated, edges)
    report["moe_combined"]["error_by_manipulation_type"] = {
        str(kind): {"videos": int((gated["manip_type"] == kind).sum()), "mean_alpha": round(float(gated["alpha"][gated["manip_type"] == kind].mean()), 3)}
        for kind in np.unique(gated["manip_type"])
    }
    (out_dir / "metrics.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")

    print("model | videos | real flagged | fake missed | probabilities exactly 0.5")
    for name, row in report.items():
        c = row["confusion_at_0.5"]
        print(f"{name} | {row['videos']} | {c['real_flagged']} of {c['real_correct'] + c['real_flagged']} | {c['fake_missed']} of {c['fake_missed'] + c['fake_caught']} | {row['probabilities_exactly_0.5']}")
    print("\nmoe_combined, error rate by gate weight")
    print("alpha | videos | fake share | error rate")
    for row in report["moe_combined"]["error_by_alpha"]:
        print(f"{row['alpha_from']:.1f} to {row['alpha_to']:.1f} | {row['videos']} | {row['fake_share']:.3f} | {row['error_rate']:.4f}")


if __name__ == "__main__":
    main()
