from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import load_reference


def counts(records: list) -> dict:
    real = [r for r in records if r[1] == 0]
    fake = [r for r in records if r[1] == 1]
    return {
        "videos": len(records), "real": len(real), "fake": len(fake),
        "segments": sum(len(r[0]) for r in records), "subjects": len({r[2] for r in records}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count videos, segments and subjects per dataset and per partition of the combined subject-level split.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference()
    seed = args.seed if args.seed is not None else repro["split"]["seed"]

    table: dict[str, dict] = {}
    for name in tqdm(config.dataset_names(), desc="datasets"):
        records = dataset_records(reference, config, name)
        train, val, test = reference.subject_split(records, repro["split"]["val_ratio"], seed)
        table[name] = {"all": counts(records), "train": counts(train), "validation": counts(val), "test": counts(test)}

    out = config.paths.results_dir("dataset_partitions")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"partitions_seed{seed}.yaml").write_text(yaml.safe_dump({"seed": seed, "val_ratio": repro["split"]["val_ratio"], "datasets": table}, sort_keys=False), encoding="utf-8")
    print("dataset | partition | videos | real | fake | segments | subjects")
    for name, parts in table.items():
        for part, c in parts.items():
            print(f"{name} | {part} | {c['videos']} | {c['real']} | {c['fake']} | {c['segments']} | {c['subjects']}")


if __name__ == "__main__":
    main()
