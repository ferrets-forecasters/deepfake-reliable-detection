"""Leave-one-out FakeAVCeleb scored without the real videos that HiDF holds a copy of (audit D1).

HiDF is in the training pool of that run, so a real FakeAVCeleb test video with an identical copy in HiDF was
seen in training. The fake videos stay; only those reals are removed from the test, in four variants."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS = ROOT / "results" / "inferences" / "moe_leave_fakeavceleb_retrain_on_fakeavceleb" / "predictions.csv"
DUPLICATES = ROOT / "results" / "analyses" / "split_leakage_audit" / "duplicates_hidf_fakeavceleb.csv"
OUT = ROOT / "results" / "metrics" / "leave_one_out_fakeavceleb_dedup" / "metrics.yaml"
VARIANTS = {
    "all_videos": None,
    "without_copies_in_hidf_train": {"train"},
    "without_copies_in_hidf_train_or_validation": {"train", "validation"},
    "without_all_copies": {"train", "validation", "test"},
}


def metrics(frame: pd.DataFrame) -> dict:
    y, p = frame["label"].to_numpy(), frame["p_fake_best_f1"].to_numpy()
    return {"auc": roc_auc_score(y, p), "ap": average_precision_score(y, p), "macro_f1": f1_score(y, p >= 0.5, average="macro")}


def interval(frame: pd.DataFrame, resamples: int, seed: int) -> list[float]:
    rng, aucs = np.random.default_rng(seed), []
    y, p = frame["label"].to_numpy(), frame["p_fake_best_f1"].to_numpy()
    for _ in range(resamples):
        pick = rng.integers(0, len(y), len(y))
        if y[pick].min() != y[pick].max():
            aucs.append(roc_auc_score(y[pick], p[pick]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_csv(PREDICTIONS)
    copies = pd.read_csv(DUPLICATES, dtype=str)
    result = {}
    for name, partitions in VARIANTS.items():
        drop = set() if partitions is None else {"fac_" + s for s in copies.loc[copies["hidf_partition"].isin(partitions), "fakeavceleb_sample"]}
        kept = frame[~((frame["label"] == 0) & frame["subject"].isin(drop))]
        row = {k: round(float(v), 4) for k, v in metrics(kept).items()}
        row.update(real=int((kept["label"] == 0).sum()), fake=int((kept["label"] == 1).sum()),
                   auc_interval_95=[round(v, 4) for v in interval(kept, args.resamples, args.seed)])
        result[name] = row
        print(name, row)
    result["copies"] = {"identical_pairs": int(len(copies)), **{k: int(v) for k, v in copies["hidf_partition"].value_counts().items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
