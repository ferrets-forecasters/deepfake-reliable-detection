from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.face_crop import MultiFaceTracker

COLORS = [(230, 40, 40), (40, 90, 230), (40, 170, 70), (230, 160, 20), (150, 60, 200)]


def read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise ValueError(f"could not read frame {index}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MultiFaceTracker on the tiled videos and report how many tracks it finds and where.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="fakeavceleb_tiled")
    parser.add_argument("--limit", type=int, default=0, help="0 = every video")
    parser.add_argument("--grid", type=int, default=12, help="videos drawn in the picture")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.paths.datasets_root / args.dataset
    with open(root / "labels" / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    tracker = MultiFaceTracker(detector_fps=config.preprocessing.video.face_detector_fps)
    out_dir = config.paths.analysis_dir("multiface_tracks")
    out_dir.mkdir(parents=True, exist_ok=True)

    report, tiles = [], []
    for row in tqdm(rows, desc="videos"):
        cap = cv2.VideoCapture(str(root / row["raw_path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        last = int(min(2.0 * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        indices = np.linspace(0, last, 16).round().astype(int).tolist()
        try:
            tracks = tracker.tracks_for_indices(lambda i: read_frame(cap, i), indices, fps)
        except ValueError:
            tracks = []
        centers = sorted(float(np.mean([(b[0] + b[2]) / 2 for b in t.detections.values()])) / width for t in tracks)
        report.append({
            "sample_id": row["sample_id"], "fake_side": row["fake_side"], "tracks": len(tracks),
            "left_track": any(c < 0.5 for c in centers), "right_track": any(c >= 0.5 for c in centers),
        })
        if len(tiles) < args.grid:
            frame = read_frame(cap, indices[len(indices) // 2]).copy()
            for t, track in enumerate(tracks):
                x0, y0, x1, y1 = (int(v) for v in track.boxes_for_indices([indices[len(indices) // 2]])[indices[len(indices) // 2]])
                cv2.rectangle(frame, (x0, y0), (x1, y1), COLORS[t % len(COLORS)], 3)
            cv2.putText(frame, f"{row['fake_side']} fake, {len(tracks)} tracks", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(frame)
        cap.release()

    if tiles:
        cols = 3
        height, width = tiles[0].shape[:2]
        sheet = np.zeros((((len(tiles) + cols - 1) // cols) * height, cols * width, 3), np.uint8)
        for i, tile in enumerate(tiles):
            r, c = divmod(i, cols)
            sheet[r * height:(r + 1) * height, c * width:(c + 1) * width] = cv2.resize(tile, (width, height))
        cv2.imwrite(str(out_dir / "tracks.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    with open(out_dir / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    counts = {}
    for r in report:
        counts[r["tracks"]] = counts.get(r["tracks"], 0) + 1
    both = sum(r["tracks"] == 2 and r["left_track"] and r["right_track"] for r in report)
    print(f"{len(report)} videos | tracks per video: {dict(sorted(counts.items()))} | exactly one track per side: {both}")
    print(f"picture and table in {out_dir}")


if __name__ == "__main__":
    main()
