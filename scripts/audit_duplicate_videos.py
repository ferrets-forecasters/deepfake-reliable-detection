from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import dataset_records
from deepfake_detection.reference import load_reference_repo


UNREADABLE: list[str] = []


def sequence(files: list[Path]) -> tuple[np.ndarray, str] | None:
    """Per-segment video and audio vectors of one video, and the path it was extracted from; None when a file cannot be read."""
    parts, orig = [], ""
    for path in files:
        try:
            # allow_pickle is needed for the stored orig_path string; these .npz files were written by this project's own extraction
            with np.load(path, allow_pickle=True) as d:
                parts.append(np.concatenate([d["video_feat"], d["audio_feat"]]))
                orig = str(d["orig_path"])
        except Exception:
            UNREADABLE.append(str(path))
            return None
    return np.stack(parts), orig


def first_frame(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return cv2.resize(frame, (64, 64)).astype(np.float32) if ok else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Find videos of one dataset that are the same video as a video of another dataset: identical segment vectors, with the original path of both sides and the partition of both in the subject-level split.")
    parser.add_argument("--a", default="hidf")
    parser.add_argument("--b", default="fakeavceleb")
    parser.add_argument("--label", type=int, default=0, help="0 for real videos")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="largest absolute difference of any segment vector entry to call two videos identical")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    seed = args.seed if args.seed is not None else repro["split"]["seed"]

    sides = {}
    for dataset in (args.a, args.b):
        records = [r for r in dataset_records(reference, config, dataset) if r[1] == args.label]
        train, val, test = reference.subject_split(dataset_records(reference, config, dataset), repro["split"]["val_ratio"], seed)
        part = {}
        for name, items in (("train", train), ("validation", val), ("test", test)):
            for r in items:
                part[r[2]] = name
        sides[dataset] = [(r[0], r[2], part[r[2]]) for r in records]

    b_vectors = []
    for files, subject, partition in tqdm(sides[args.b], desc=args.b):
        loaded = sequence(files)
        if loaded is None:
            continue
        vectors, orig = loaded
        b_vectors.append((files[0].name.rsplit("_seg", 1)[0], subject, partition, vectors, orig))
    first_b = np.stack([v[3][0] for v in b_vectors])

    rows = []
    for files, subject, partition in tqdm(sides[args.a], desc=args.a):
        loaded = sequence(files)
        if loaded is None:
            continue
        vectors, orig = loaded
        difference = np.abs(first_b - vectors[0]).max(axis=1)
        for index in np.where(difference <= args.tolerance)[0]:
            name_b, subject_b, partition_b, vectors_b, orig_b = b_vectors[index]
            n = min(len(vectors), len(vectors_b))
            same_all = bool(np.abs(vectors[:n] - vectors_b[:n]).max() <= args.tolerance)
            rows.append({
                f"{args.a}_sample": files[0].name.rsplit("_seg", 1)[0], f"{args.a}_subject": subject, f"{args.a}_partition": partition, f"{args.a}_orig_path": orig,
                f"{args.b}_sample": name_b, f"{args.b}_subject": subject_b, f"{args.b}_partition": partition_b, f"{args.b}_orig_path": orig_b,
                "segments_compared": n, "all_segments_identical": same_all, "max_abs_difference": float(np.abs(vectors[:n] - vectors_b[:n]).max()),
            })

    out = config.paths.analysis_dir("split_leakage_audit")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"duplicates_{args.a}_{args.b}.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["none"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"unreadable embedding files: {UNREADABLE}")
    matched_a = {r[f"{args.a}_sample"] for r in rows}
    print(f"{len(matched_a)} of {len(sides[args.a])} {args.a} videos have an identical {args.b} video; pairs in {path}")
    for p in ("train", "validation", "test"):
        print(p, sum(1 for r in rows if r[f"{args.a}_partition"] == p), "pairs with the", args.a, "side in", p)
    cross = sum(1 for r in rows if r[f"{args.a}_partition"] != r[f"{args.b}_partition"])
    print(f"{cross} of {len(rows)} pairs have the two sides in different partitions")


if __name__ == "__main__":
    main()
