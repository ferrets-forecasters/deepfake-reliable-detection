from __future__ import annotations

import csv

import torch

from deepfake_detection.config import Config
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import ReferenceRepo

DATASET_ORDER = ["polyglotfake", "hidf", "fakeavceleb", "celebdfv2", "ffpp"]


def load_fold_assignment(config: Config, dataset: str, folds: int, seed: int) -> dict[str, int]:
    path = config.paths.datasets_root / dataset / "labels" / f"cv_folds_k{folds}_seed{seed}.csv"
    with open(path, encoding="utf-8") as f:
        return {row["subject"]: int(row["fold"]) for row in csv.DictReader(f)}


def load_kfold_splits(reference: ReferenceRepo, config: Config, fold: int, folds: int = 5, seed: int = 42):
    """Train, validation and test records of one cross-validation fold, in the shape the old repo's
    trainer takes: fold `fold` is the test set, fold `fold + 1` the validation set and the remaining
    folds train, taken per dataset. Domain labels follow the old combined run (PolyglotFake 0, HiDF 1,
    FakeAVCeleb 2, CelebDFv2 3, FF++ 4)."""
    train, val, test, domain_ids = [], [], [], []
    for domain, name in enumerate(DATASET_ORDER):
        assignment = load_fold_assignment(config, name, folds, seed)
        validation_fold = (fold + 1) % folds
        for row in dataset_records(reference, config, name):
            assigned = assignment[row[2]]
            if assigned == fold:
                test.append(row)
            elif assigned == validation_fold:
                val.append(row)
            else:
                train.append(row)
                domain_ids.append(domain)
    return train, val, test, torch.tensor(domain_ids, dtype=torch.long), None
