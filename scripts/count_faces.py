from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.inference.multi_subject import frame_reader

COLUMNS = ["video", "width", "height", "seconds", "tracks", "largest_share", "second_share"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Count the persistent face tracks of every video in a folder: how many videos hold more than one person.")
    parser.add_argument("folder", help="folder with the videos, searched recursively for .mp4")
    parser.add_argument("--name", required=True, help="output name under results/analyses")
    parser.add_argument("--config", default=None)
    parser.add_argument("--detector-fps", type=float, default=2.0)
    parser.add_argument("--max-faces", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.analysis_dir(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "face_counts.csv"
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {r["video"] for r in csv.DictReader(f)}
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    tracker = MultiFaceTracker(args.detector_fps, max_faces=args.max_faces)
    for path in tqdm(sorted(Path(args.folder).rglob("*.mp4")), desc="videos"):
        if path.name in done:
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            tracks = tracker.tracks_for_indices(frame_reader(cap, frames), [0, frames - 1], fps)
        except ValueError:
            tracks = []
        cap.release()
        presence = sorted((len(t.detections) for t in tracks), reverse=True)
        total = max(frames / fps * args.detector_fps, 1)
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow({
                "video": path.name, "width": width, "height": height, "seconds": round(frames / fps, 1), "tracks": len(tracks),
                "largest_share": round(presence[0] / total, 2) if presence else 0.0,
                "second_share": round(presence[1] / total, 2) if len(presence) > 1 else 0.0,
            })

    with open(out_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = np.array([int(r["tracks"]) for r in rows])
    persistent = np.array([float(r["second_share"]) >= 0.5 for r in rows])
    print(f"{len(rows)} videos: no face {int((counts == 0).sum())}, one track {int((counts == 1).sum())}, two or more {int((counts >= 2).sum())}, "
          f"a second person present in at least half of the detections {int(persistent.sum())}")


if __name__ == "__main__":
    main()
