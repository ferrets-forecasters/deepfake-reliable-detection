from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import DRIVE_URL, open_remote  # noqa: E402

from deepfake_detection.config import load_config  # noqa: E402

PREDICTIONS = "moe_combined_retrain_on_combined_test_seed42"
ARCHIVES = {
    "celebdfv2": ("1ycE03-UpDrrTKfem5FSewMNf5mtHL5Cw", "Celeb-DF-v2.zip"),
    "ffpp": ("18M8bOxbPT_d6Zii7Ah2HAV7XgG0_piY_", "FaceForensics++ Dataset (C23).zip"),
}
FFPP_METHODS = ("deepfakes", "face2face", "faceshifter", "faceswap", "neuraltextures")


def load_scores(config, dataset: str) -> dict[str, dict]:
    path = config.paths.predictions_dir(PREDICTIONS) / "predictions.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] == dataset:
                out[re.sub(r"_seg\d+$", "", Path(row["video"]).stem)] = {"label": int(row["label"]), "p_fake": float(row["p_fake_best_f1"])}
    return out


def candidate_pairs(dataset: str, scores: dict, members: dict) -> list[dict]:
    """Own-test fakes whose real clip of the same scene is also in the own test split and in the archive. Celeb-DF: fake idX_idY_N
    is the real idX_N with the face of idY. FF++: fake <method>_N_M is the original N with the face of M."""
    out = []
    for key, s in scores.items():
        if s["label"] != 1:
            continue
        if dataset == "celebdfv2":
            m = re.match(r"^(id\d+)_(id\d+)_(\d+)$", key)
            real_key, fake_member, real_member, group = (f"{m[1]}_{m[3]}", members.get(("", key)), members.get(("", f"{m[1]}_{m[3]}")), "celebdfv2") if m else (None,) * 4
        else:
            m = re.match(r"^([a-z0-9]+)_(\d{3})_(\d{3})$", key)
            ok = m and m[1] in FFPP_METHODS
            real_key = m[2] if ok else None
            fake_member = members.get((m[1], f"{m[2]}_{m[3]}")) if ok else None
            real_member = members.get(("original", m[2])) if ok else None
            group = m[1] if ok else None
        if real_key and real_key in scores and fake_member and real_member:
            out.append({"fake_key": key, "real_key": real_key, "fake_member": fake_member, "real_member": real_member, "group": group,
                        "failure": s["p_fake"] < 0.5 or scores[real_key]["p_fake"] >= 0.5})
    return out


def member_index(dataset: str, config) -> dict:
    index = {}
    for line in (config.paths.datasets_root.parent / "temp" / "lists" / f"{dataset}_members.txt").read_text(encoding="utf-8").splitlines():
        name = line.split("\t")[1]
        parts = Path(name).parts
        index[("" if dataset == "celebdfv2" else parts[1].lower(), Path(name).stem)] = name
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Example real and fake pairs of datasets whose archives are not on this machine: pairs of the own test split are picked, only their members are fetched from the shared Drive archives by ranged requests, and export_example_pairs.py exports frames and the manifest.")
    parser.add_argument("--datasets", nargs="+", default=["celebdfv2", "ffpp"])
    parser.add_argument("--random-pairs", type=int, default=6)
    parser.add_argument("--failure-pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = random.Random(args.seed)
    for dataset in args.datasets:
        scores = load_scores(config, dataset)
        pool = candidate_pairs(dataset, scores, member_index(dataset, config))
        rng.shuffle(pool)
        failures = [p for p in pool if p["failure"]][: args.failure_pairs + 2]
        randoms, used = [], set()
        for p in (q for q in pool if not q["failure"]):
            if p["group"] in used and len(used) < len({q["group"] for q in pool}):
                continue
            used.add(p["group"])
            randoms.append(p)
            if len(randoms) == args.random_pairs + 2:
                break
        chosen = failures + randoms
        file_id, _ = ARCHIVES[dataset]
        archive = open_remote(DRIVE_URL.format(id=file_id))
        raw = config.paths.datasets_root / dataset / "raw" / "test_pairs"
        pairs = []
        for p in tqdm(chosen, desc=f"{dataset} fetch"):
            paths = []
            for member in (p["real_member"], p["fake_member"]):
                target = raw / member
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, open(target, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                paths.append(str(target))
            pairs += ["--pair", *paths]
        count = args.random_pairs + args.failure_pairs
        subprocess.run([sys.executable, str(Path(__file__).parent / "export_example_pairs.py"), "--dataset", dataset, "--count", str(count),
                        "--failures", str(args.failure_pairs), *pairs], check=True)


if __name__ == "__main__":
    main()
