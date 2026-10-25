from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

from deepfake_detection.config import load_config

COLUMN = "p_fake_best_f1"


def split_folders() -> list[tuple[str, str, str]]:
    seeds = [("seed 42", "moe_combined_retrain_on_combined_test_seed42", "concat_combined_retrain_on_combined_test_seed42")]
    seeds += [(f"seed {s}", f"moe_combined_seed{s}_on_combined_test_seed{s}", f"concat_combined_seed{s}_on_combined_test_seed{s}") for s in (43, 44)]
    folds = [(f"fold {i}", f"moe_cv_fold{i}_on_cv_fold{i}_test", f"concat_cv_fold{i}_on_cv_fold{i}_test") for i in range(5)]
    return seeds + folds


def main() -> None:
    parser = argparse.ArgumentParser(description="McNemar test of the gated model against the static model on each of the eight test splits (three seeds, five cross-validation folds), with the sign of the F1 and AUC difference, and a sign test over the splits. Saved predictions only, no model is run.")
    parser.add_argument("--name", default="gate_vs_static")
    args = parser.parse_args()
    config = load_config()
    rows = []
    for name, gate_run, static_run in split_folders():
        gate = pd.read_csv(config.paths.predictions_dir(gate_run) / "predictions.csv")
        static = pd.read_csv(config.paths.predictions_dir(static_run) / "predictions.csv")
        both = gate.set_index("video")[["label", COLUMN]].join(static.set_index("video")[[COLUMN]], rsuffix="_static", how="inner")
        label, pg, ps = both.label.to_numpy(), both[COLUMN].to_numpy(), both[COLUMN + "_static"].to_numpy()
        right_g, right_s = (pg >= 0.5) == (label == 1), (ps >= 0.5) == (label == 1)
        only_g, only_s = int((right_g & ~right_s).sum()), int((~right_g & right_s).sum())
        rows.append({"split": name, "videos": len(both), "only_gate": only_g, "only_static": only_s,
                     "mcnemar_p": binomtest(only_g, only_g + only_s, 0.5).pvalue,
                     "f1_gate": f1_score(label, pg >= 0.5, average="macro"), "f1_static": f1_score(label, ps >= 0.5, average="macro"),
                     "auc_gate": roc_auc_score(label, pg), "auc_static": roc_auc_score(label, ps)})
    table = pd.DataFrame(rows)
    out_dir = config.paths.analysis_dir(args.name)
    table.to_csv(out_dir / "per_split.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    for label, wins in (("McNemar count (only gate > only static)", (table.only_gate > table.only_static)),
                        ("macro-F1 at 0.5 (gate higher)", (table.f1_gate.round(4) > table.f1_static.round(4))),
                        ("AUC (gate higher)", (table.auc_gate.round(4) > table.auc_static.round(4)))):
        n = int(wins.sum())
        print(f"{label}: {n} of {len(table)} splits, sign test p = {binomtest(n, len(table), 0.5).pvalue:.3f}")


if __name__ == "__main__":
    main()
