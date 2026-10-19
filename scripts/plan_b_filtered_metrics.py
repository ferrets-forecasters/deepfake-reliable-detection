from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT
from deepfake_detection.data.partners import partner_keys
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import load_reference

RUNS = {42: ("moe_combined_retrain", "concat_combined_retrain"), 43: ("moe_combined_seed43", "concat_combined_seed43"), 44: ("moe_combined_seed44", "concat_combined_seed44")}


def read_predictions(config, run: str, seed: int) -> dict[tuple[str, str], tuple[int, float, float]]:
    path = config.paths.predictions_dir(f"{run}_on_combined_test_seed{seed}") / "predictions.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            alpha = float(row["alpha_best_f1"]) if row["alpha_best_f1"] not in ("", "nan") else float("nan")
            out[(row["dataset"], re.sub(r"_seg\d+$", "", Path(row["video"]).stem))] = (int(float(row["label"])), float(row["p_fake_best_f1"]), alpha)
    return out


def metrics(items: list[tuple[int, float, float]]) -> dict:
    if not items:
        return {"videos": 0}
    labels = np.array([i[0] for i in items])
    probs = np.array([i[1] for i in items])
    out = {"videos": len(items), "real": int((labels == 0).sum()), "fake": int((labels == 1).sum()), "macro_f1": round(float(f1_score(labels, probs >= 0.5, average="macro")), 4)}
    if len(set(labels.tolist())) == 2:
        out["auc"] = round(float(roc_auc_score(labels, probs)), 4)
        out["ap"] = round(float(average_precision_score(labels, probs)), 4)
    alphas = np.array([i[2] for i in items])
    if not np.isnan(alphas).all():
        out["alpha_mean"] = round(float(np.nanmean(alphas)), 4)
    return out


def paired_difference(a: list[tuple[int, float, float]], b: list[tuple[int, float, float]], resamples: int = 1000, seed: int = 42) -> dict:
    labels = np.array([x[0] for x in a])
    pa, pb = np.array([x[1] for x in a]), np.array([x[1] for x in b])
    rng = np.random.default_rng(seed)
    out = {}
    for name, fn in (("macro_f1", lambda y, p: f1_score(y, p >= 0.5, average="macro")), ("auc", roc_auc_score)):
        base = fn(labels, pa) - fn(labels, pb)
        draws = []
        for _ in range(resamples):
            idx = rng.integers(0, len(labels), len(labels))
            if len(set(labels[idx].tolist())) < 2:
                continue
            draws.append(fn(labels[idx], pa[idx]) - fn(labels[idx], pb[idx]))
        out[name] = {"difference": round(float(base), 4), "ci95": [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Metrics of the combined test split from the saved predictions, before and after removing the test videos that share a video or an identity with a train or validation video. No training.")
    parser.add_argument("--seed", type=int, default=42, choices=[42, 43, 44])
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference()
    audit = config.paths.analysis_dir("split_leakage_audit")

    subject_part: dict[tuple[str, str], str] = {}
    sample_part: dict[tuple[str, str], str] = {}
    sample_subject: dict[tuple[str, str], str] = {}
    for dataset in config.dataset_names():
        train, val, test = reference.subject_split(dataset_records(reference, config, dataset), repro["split"]["val_ratio"], args.seed)
        for part, items in (("train", train), ("validation", val), ("test", test)):
            for r in items:
                name = r[0][0].stem.rsplit("_seg", 1)[0]
                subject_part[(dataset, r[2])] = part
                sample_part[(dataset, name)] = part
                sample_subject[(dataset, name)] = r[2]

    excluded: dict[str, set[tuple[str, str]]] = {"D1 HiDF reals that are FakeAVCeleb reals": set(), "D2 FakeAVCeleb identity also outside test": set(), "D5 PolyglotFake fake with source real outside test": set()}
    test_samples = [k for k, p in sample_part.items() if p == "test"]
    by_subject: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in test_samples:
        by_subject.setdefault((key[0], sample_subject[key]), []).append(key)

    with open(audit / "duplicates_hidf_fakeavceleb.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ph = subject_part.get(("hidf", row["hidf_subject"]))
            pf = subject_part.get(("fakeavceleb", row["fakeavceleb_subject"]))
            if ph == "test" and pf != "test":
                excluded["D1 HiDF reals that are FakeAVCeleb reals"].update(by_subject.get(("hidf", row["hidf_subject"]), []))
            if pf == "test" and ph != "test":
                excluded["D1 HiDF reals that are FakeAVCeleb reals"].update(by_subject.get(("fakeavceleb", row["fakeavceleb_subject"]), []))

    root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    identity_subjects: dict[str, set[str]] = {}
    for path in root.glob("*/*/*/*/*.mp4"):
        identity_subjects.setdefault(path.parts[-2], set()).add("fac_" + path.stem.split("_")[0])
    for subjects in identity_subjects.values():
        parts = {subject_part.get(("fakeavceleb", s)) for s in subjects} - {None}
        if "test" in parts and parts != {"test"}:
            for s in subjects:
                if subject_part.get(("fakeavceleb", s)) == "test":
                    excluded["D2 FakeAVCeleb identity also outside test"].update(by_subject.get(("fakeavceleb", s), []))

    for (dataset, name) in test_samples:
        if dataset != "polyglotfake":
            continue
        for _, key in partner_keys(dataset, name):
            other = sample_part.get((dataset, key))
            if other in ("train", "validation"):
                excluded["D5 PolyglotFake fake with source real outside test"].add((dataset, name))

    gated, static = (read_predictions(config, run, args.seed) for run in RUNS[args.seed])
    keys = sorted(set(gated) & set(static))
    union = set().union(*excluded.values())
    report = {"seed": args.seed, "test_videos": len(keys), "excluded": {k: len(v & set(keys)) for k, v in excluded.items()}, "excluded_union": len(union & set(keys)),
              "tests": {}}
    for label, keep in (("all test videos", keys), ("without the excluded videos", [k for k in keys if k not in union])):
        part = {"gated": metrics([gated[k] for k in keep]), "static": metrics([static[k] for k in keep]),
                "gated_minus_static": paired_difference([gated[k] for k in keep], [static[k] for k in keep])}
        part["per_dataset"] = {ds: {"gated": metrics([gated[k] for k in keep if k[0] == ds]), "static": metrics([static[k] for k in keep if k[0] == ds])} for ds in config.dataset_names()}
        report["tests"][label] = part

    out = audit / f"plan_b_seed{args.seed}.yaml"
    out.write_text(yaml.safe_dump(report, sort_keys=False, width=140), encoding="utf-8")
    print(yaml.safe_dump({k: report[k] for k in ("seed", "test_videos", "excluded", "excluded_union")}, sort_keys=False))
    for label, part in report["tests"].items():
        print(f"{label}: gated {part['gated']} | static {part['static']}")
        print("  gated minus static:", part["gated_minus_static"])
        for ds, v in part["per_dataset"].items():
            print(f"  {ds}: gated {v['gated']} | static {v['static']}")


if __name__ == "__main__":
    main()
