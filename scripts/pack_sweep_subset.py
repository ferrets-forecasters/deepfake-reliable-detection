from __future__ import annotations

import argparse
import csv
import random
import tarfile
from pathlib import Path

from tqdm import tqdm

from deepfake_detection.config import load_config

MANIFEST_COLUMNS = ["video_id", "dataset", "label", "condition", "path", "bytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Packs a subset of the compression sweep for a pod: per dataset and label, `--per-class` videos that the paper model has scored under every condition, with a manifest of relative paths.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--name", default="compression_sweep")
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.paths.datasets_root / args.name
    with open(root / "labels" / "manifest.csv", encoding="utf-8", newline="") as f:
        manifest = list(csv.DictReader(f))
    with open(config.paths.predictions_dir(f"{args.model}_on_{args.name}") / "predictions.csv", encoding="utf-8", newline="") as f:
        scored = {}
        for r in csv.DictReader(f):
            if r["p_fake"] != "":
                scored.setdefault(r["video_id"], set()).add(r["condition"])
    conditions = {r["condition"] for r in manifest}
    complete = sorted({r["video_id"] for r in manifest if scored.get(r["video_id"]) == conditions})

    groups = {}
    for r in manifest:
        if r["condition"] == "original" and r["video_id"] in complete:
            groups.setdefault((r["dataset"], r["label"]), []).append(r["video_id"])
    rng = random.Random(args.seed)
    chosen = {v for ids in groups.values() for v in rng.sample(ids, min(args.per_class, len(ids)))}
    rows = [r for r in manifest if r["video_id"] in chosen]

    out = args.out or config.paths.datasets_root.parent / "temp" / "sweep_subset.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    base = Path("gemastik") / "datasets" / args.name
    kept = []
    with tarfile.open(out, "w:gz") as tar:
        for r in tqdm(rows, desc="pack", unit="clip"):
            relative = Path("raw") / r["condition"] / f"{r['video_id']}.mp4"
            tar.add(r["path"], arcname=base / relative)
            kept.append({**r, "path": relative.as_posix()})
        manifest_path = out.with_name("subset_manifest.csv")
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows({k: r[k] for k in MANIFEST_COLUMNS} for r in kept)
        tar.add(manifest_path, arcname=base / "labels" / "manifest.csv")
    manifest_path.unlink()
    print(f"{out} {out.stat().st_size / 2**20:.0f} MB, {len(chosen)} videos, {len(rows)} clips")


if __name__ == "__main__":
    main()
