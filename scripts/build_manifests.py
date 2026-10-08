from __future__ import annotations

import argparse

from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data import (
    SegmentRecord,
    build_celebdfv2_manifest,
    build_fakeavceleb_manifest,
    build_ffpp_manifest,
    subject_level_split,
    write_manifest,
)

BUILDERS = {
    "ffpp": build_ffpp_manifest,
    "celebdfv2": build_celebdfv2_manifest,
    "fakeavceleb": build_fakeavceleb_manifest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segment-level manifests from raw downloaded datasets.")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--datasets", nargs="+", default=list(BUILDERS), help="Only ffpp/celebdfv2/fakeavceleb have a dedicated "
        "builder so far; hidf and polyglotfake need build_generic_manifest wired in once their raw layout is known."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    all_records: list[SegmentRecord] = []

    for name in tqdm(args.datasets, desc="datasets"):
        builder = BUILDERS[name]
        raw_dir = config.paths.raw_dir / name
        records = builder(raw_dir, config.preprocessing.segment_seconds)
        assignment = subject_level_split([r.subject_id for r in records], config.split, config.seed)
        for record in records:
            record.split = assignment[record.subject_id]
        all_records.extend(records)
        write_manifest(records, config.paths.manifests_dir / f"{name}.csv")
        print(f"{name}: {len(records)} segments, {len(set(assignment.values()))} split groups")

    write_manifest(all_records, config.paths.manifests_dir / "combined.csv")


if __name__ == "__main__":
    main()
