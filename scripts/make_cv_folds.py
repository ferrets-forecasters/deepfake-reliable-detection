from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import yaml

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import load_reference_repo


def assign_folds(records: list, folds: int, seed: int) -> dict[str, int]:
    """Subject-level folds balanced on the number of videos. Subjects are taken largest first, ties in a seeded
    random order, and each goes to the fold with the fewest videos so far, then the fewest real videos."""
    videos, real = defaultdict(int), defaultdict(int)
    for row in records:
        videos[row[2]] += 1
        real[row[2]] += int(row[1]) == 0
    subjects = sorted(videos)
    random.Random(seed).shuffle(subjects)
    subjects.sort(key=lambda s: -videos[s])
    totals, reals = [0] * folds, [0] * folds
    assignment = {}
    for subject in subjects:
        fold = min(range(folds), key=lambda f: (totals[f], reals[f]))
        assignment[subject] = fold
        totals[fold] += videos[subject]
        reals[fold] += real[subject]
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the subject-level k-fold assignment of every dataset to datasets/<name>/labels/cv_folds_k<k>_seed<seed>.csv and print the size and class balance of each fold.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])

    print("dataset | fold | subjects | videos | real videos | fake share")
    for name in config.dataset_names():
        records = dataset_records(reference, config, name)
        assignment = assign_folds(records, args.folds, args.seed)
        out = config.paths.datasets_root / name / "labels" / f"cv_folds_k{args.folds}_seed{args.seed}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "fold"])
            writer.writerows(sorted(assignment.items()))
        rows = defaultdict(lambda: [set(), 0, 0])
        for row in records:
            entry = rows[assignment[row[2]]]
            entry[0].add(row[2])
            entry[1] += 1
            entry[2] += int(row[1]) == 0
        assert sum(len(v[0]) for v in rows.values()) == len(assignment), "a subject is in two folds"
        assert sum(v[1] for v in rows.values()) == len(records), "a video is in no fold"
        for fold in sorted(rows):
            subjects, videos, real = rows[fold]
            print(f"{name} | {fold} | {len(subjects)} | {videos} | {real} | {1 - real / videos:.3f}")


if __name__ == "__main__":
    main()
