from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from deepfake_detection.config import load_config


def read_manifest(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def tiled_summary(predictions: Path) -> None:
    """Scores of the first 2-second segment of the tiled videos: single face against the per-face maximum."""
    with open(predictions, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    plans: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        plans[row["sample_id"].rsplit("_", 1)[0]].append(row)
    fake_faces, real_faces = [], []
    print("plan | n | single face flags | per-face maximum flags | fake face mean p_fake | real face mean p_fake")
    for plan, group in plans.items():
        side = group[0]["fake_side"]
        other = {"left": "right", "right": "left"}.get(side)
        real = [float(r[f"{other}_p_fake"]) for r in group] if other else [float(r[k]) for r in group for k in ("left_p_fake", "right_p_fake")]
        fake = [float(r[f"{side}_p_fake"]) for r in group] if other else []
        fake_faces += fake
        real_faces += real
        single = np.mean([float(r["single_p_fake"]) >= 0.5 for r in group])
        multi = np.mean([float(r["multi_p_fake"]) >= 0.5 for r in group])
        print(f"{plan} | {len(group)} | {single:.2f} | {multi:.2f} | {np.mean(fake) if fake else float('nan'):.3f} | {np.mean(real):.3f}")
    labels = np.array([int(r["label"]) for r in rows])
    for column in ("single_p_fake", "multi_p_fake"):
        scores = np.array([float(r[column]) for r in rows])
        print(f"{column}: video AUC {roc_auc_score(labels, scores):.4f}, accuracy at 0.5 {((scores >= 0.5) == labels).mean():.4f}")
    faces = np.r_[np.ones(len(fake_faces)), np.zeros(len(real_faces))]
    print(f"fake face against real face: AUC {roc_auc_score(faces, np.r_[fake_faces, real_faces]):.4f} ({len(fake_faces)} fake faces, {len(real_faces)} real faces)")


def conversation_summary(results_dir: Path, manifest: dict[str, dict]) -> None:
    """Per-person verdicts of the turn-taking videos, each person scored on their own speaker's audio."""
    plans: dict[str, list[float]] = defaultdict(list)
    labels, video_scores, fake_faces, real_faces, linked, faces = [], [], [], [], 0, 0
    for path in sorted(results_dir.glob("*.json")):
        result, meta = json.loads(path.read_text(encoding="utf-8")), manifest[path.stem]
        if not result["subjects"]:
            continue
        plans[path.stem.rsplit("_", 1)[0]].append(result["video_p_fake"])
        labels.append(int(meta["label"]))
        video_scores.append(result["video_p_fake"])
        people = sorted(result["subjects"], key=lambda s: s["center_x"])
        if len(people) != 2:
            continue
        faces += 2
        linked += sum(p["speaker"] is not None for p in people)
        for position, person in zip(("left", "right"), people):
            (fake_faces if meta["fake_side"] == position else real_faces).append(person["p_fake"])
    print("plan | n | mean video p_fake | flagged at 0.5")
    for plan, scores in plans.items():
        print(f"{plan} | {len(scores)} | {np.mean(scores):.3f} | {np.mean(np.array(scores) >= 0.5):.2f}")
    print(f"video AUC {roc_auc_score(labels, video_scores):.4f}")
    faces_y = np.r_[np.ones(len(fake_faces)), np.zeros(len(real_faces))]
    print(f"fake face against real face: AUC {roc_auc_score(faces_y, np.r_[fake_faces, real_faces]):.4f} ({len(fake_faces)} fake faces, {len(real_faces)} real faces)")
    print(f"people linked to a speaker: {linked} of {faces}")


def control_summary(reencoded_dir: Path, original_dir: Path, manifest: dict[str, dict]) -> None:
    """Real and fake single clips, original against the same clip after the tiling's ffmpeg scale and encode."""
    pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample_id, meta in manifest.items():
        source = Path(meta["source_clip"].replace("\\", "/"))
        re = reencoded_dir / f"{Path(meta['raw_path'].replace(chr(92), '/')).parent.name}_{sample_id}.json"
        orig = original_dir / f"{source.parent.name}_{source.stem}.json"
        if re.exists() and orig.exists():
            pairs[meta["kind"]].append((json.loads(orig.read_text())["video_p_fake"], json.loads(re.read_text())["video_p_fake"]))
    print("kind | pairs | original mean p_fake | flagged | re-encoded mean p_fake | flagged")
    for kind, values in pairs.items():
        original, re = np.array(values).T
        print(f"{kind} | {len(values)} | {original.mean():.3f} | {(original >= 0.5).mean():.2f} | {re.mean():.3f} | {(re >= 0.5).mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summaries of the multi-subject runs: tiled set, conversation set, re-encoding control.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--tiled", help="dataset name of a tiled set scored by score_tiled_set.py")
    parser.add_argument("--conversation", help="dataset name of a conversation set scored by demo.py")
    parser.add_argument("--control", help="dataset name of the re-encoded control, scored by infer_multi_subject.py")
    parser.add_argument("--original", help="name given to infer_multi_subject.py for the original clips of the control")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.paths.datasets_root
    if args.tiled:
        tiled_summary(config.paths.predictions_dir(f"{args.model}_on_{args.tiled}") / "predictions.csv")
    if args.conversation:
        conversation_summary(config.paths.predictions_dir(f"{args.model}_on_{args.conversation}"), read_manifest(root / args.conversation / "labels" / "manifest.csv"))
    if args.control and args.original:
        control_summary(config.paths.predictions_dir(f"{args.model}_on_{args.control}"), config.paths.predictions_dir(f"{args.model}_on_{args.original}"),
                        read_manifest(root / args.control / "labels" / "manifest.csv"))


if __name__ == "__main__":
    main()
