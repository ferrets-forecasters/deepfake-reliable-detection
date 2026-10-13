from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from build_tiled_set import fake_side
from deepfake_detection.config import load_config
from deepfake_detection.data.tiling import conversation_pair, index_clips, pick_pairs

# left kind, right kind. The left person talks first, then the right person.
PLANS = [("real", "real"), ("fake_av", "real"), ("real", "fake_av"), ("fake_v", "real"), ("real", "fake_v")]
COLUMNS = ["sample_id", "raw_path", "left_clip", "right_clip", "left_kind", "right_kind", "fake_side", "label"]


def duration_seconds(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
    cap.release()
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Build turn-taking two-person videos from FakeAVCeleb clips: each person talks in their own turn, with known labels.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-plan", type=int, default=8)
    parser.add_argument("--turn-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=224, help="frame height; the FakeAVCeleb clips are 224x224, so the default keeps the original pixels")
    parser.add_argument("--lossless", action="store_true", help="write FFV1 and PCM in .mkv, no lossy re-encoding")
    parser.add_argument("--ffmpeg", default=r"C:\tools\ffmpeg\bin\ffmpeg.exe")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    source_root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    out_root = config.paths.datasets_root / (args.name or ("fakeavceleb_conversation_lossless" if args.lossless else "fakeavceleb_conversation"))
    raw_out, labels_out = out_root / "raw", out_root / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    index = index_clips(source_root)
    rows = []
    for plan_number, (left_kind, right_kind) in enumerate(PLANS):
        candidates = pick_pairs(index, left_kind, right_kind, args.per_plan * 6, args.seed + plan_number)
        pairs = [(l, r) for l, r in candidates if min(duration_seconds(l), duration_seconds(r)) >= args.turn_seconds][: args.per_plan]
        for pair_number, (left, right) in enumerate(tqdm(pairs, desc=f"{left_kind}+{right_kind}")):
            sample_id = f"{left_kind}_{right_kind}_{pair_number:03d}"
            out = raw_out / f"{sample_id}{'.mkv' if args.lossless else '.mp4'}"
            if not out.exists():
                conversation_pair(Path(args.ffmpeg), left, right, out, args.turn_seconds, args.height, args.lossless)
            side = fake_side(left_kind, right_kind)
            rows.append({
                "sample_id": sample_id, "raw_path": str(out.relative_to(out_root)),
                "left_clip": str(left.relative_to(source_root)), "right_clip": str(right.relative_to(source_root)),
                "left_kind": left_kind, "right_kind": right_kind, "fake_side": side, "label": int(side != "none"),
            })

    with open(labels_out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} conversation videos in {raw_out}")


if __name__ == "__main__":
    main()
