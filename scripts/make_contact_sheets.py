from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def sheet(path: Path, step: float, columns: int, width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    tiles = []
    for t in np.arange(0, duration, step):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        height = int(frame.shape[0] * width / frame.shape[1])
        tile = cv2.resize(frame, (width, height))
        cv2.putText(tile, f"{int(t // 60)}:{int(t % 60):02d}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i:i + columns]
        row += [np.zeros_like(tiles[0])] * (columns - len(row))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="One contact sheet per video: a frame every STEP seconds with its time, to find by eye the stretch of a news segment that shows the viral clip.")
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step", type=float, default=4.0)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--width", type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.videos:
        cv2.imwrite(str(args.out / f"{path.stem}.jpg"), sheet(path, args.step, args.columns, args.width), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
