from __future__ import annotations

import argparse
import collections
import csv
import re
from pathlib import Path

import yaml

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import load_reference_repo

FAKE_NAME = re.compile(r"^fake_(?P<source>[a-z]{2}_\d+)_to_(?P<target>[a-z]{2})_(?P<voice>.+)$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Where the fakes of PolyGlotFake and their source real videos fall in the subject-level split: the reals are keyed by name and the fakes by number, so the two can sit in different partitions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    seed = args.seed if args.seed is not None else repro["split"]["seed"]

    records = dataset_records(reference, config, "polyglotfake")
    train, val, test = reference.subject_split(records, repro["split"]["val_ratio"], seed)
    partition, subject = {}, {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        for record in part:
            sample = record[0][0].stem.rsplit("_seg", 1)[0]
            partition[sample], subject[sample] = name, record[2]

    rows, tally = [], collections.Counter()
    for sample, part in partition.items():
        match = FAKE_NAME.match(sample)
        if not match:
            continue
        source = match["source"]
        source_part = partition.get(source)
        rows.append({"fake": sample, "fake_subject": subject[sample], "fake_partition": part, "source_real": source,
                     "source_subject": subject.get(source, ""), "source_partition": source_part or "not in the embedded data"})
        tally[(part, source_part or "not embedded")] += 1

    out = config.paths.analysis_dir("polyglotfake_split_check")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    known = {k: v for k, v in tally.items() if k[1] != "not embedded"}
    test_known = sum(v for (p, _), v in known.items() if p == "test")
    summary = {
        "seed": seed, "fakes": len(rows),
        "source_real_in_embedded_data": sum(known.values()),
        "source_real_not_embedded": sum(v for (p, s), v in tally.items() if s == "not embedded"),
        "fake_subject_equals_source_subject": sum(1 for r in rows if r["fake_subject"] == r["source_subject"]),
        "test_fakes_with_known_source": test_known,
        "test_fakes_whose_source_is_in_train_or_validation": sum(v for (p, s), v in known.items() if p == "test" and s != "test"),
        "by_fake_partition_and_source_partition": {f"{p} / {s}": v for (p, s), v in sorted(tally.items())},
    }
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
