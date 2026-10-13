from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.tiling import index_clips, own_test_clips, pick_pairs, tile_pair

# left kind, right kind, which side the audio comes from. fake_av carries both a swapped face and cloned audio.
PLANS = [
    ("real", "real", "left"),
    ("fake_av", "real", "left"),
    ("real", "fake_av", "left"),
    ("fake_av", "real", "right"),
    ("fake_v", "real", "left"),
    ("real", "fake_v", "right"),
]
FAKE_KINDS = {"fake_av", "fake_v", "fake_a"}
COLUMNS = ["sample_id", "raw_path", "left_clip", "right_clip", "left_kind", "right_kind", "audio_from", "fake_side", "label"]


def fake_side(left_kind: str, right_kind: str) -> str:
    left, right = left_kind in FAKE_KINDS, right_kind in FAKE_KINDS
    return "both" if left and right else "left" if left else "right" if right else "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile two FakeAVCeleb clips side by side to make two-person test videos with known labels.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-plan", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=224, help="frame height; the FakeAVCeleb clips are 224x224, so the default keeps the original pixels")
    parser.add_argument("--lossless", action="store_true", help="write FFV1 and PCM in .mkv, no lossy re-encoding")
    parser.add_argument("--ffmpeg", default=r"C:\tools\ffmpeg\bin\ffmpeg.exe")
    parser.add_argument("--own-test-only", action="store_true", help="use only clips of the own seed-42 test split (labels/test_split.csv), so no clip comes from a subject a model trained on")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    source_root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    default_name = f"fakeavceleb_tiled{'_own_test' if args.own_test_only else ''}{'_lossless' if args.lossless else ''}"
    out_root = config.paths.datasets_root / (args.name or default_name)
    raw_out, labels_out = out_root / "raw", out_root / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    allowed = own_test_clips(source_root, config.paths.datasets_root / "fakeavceleb" / "labels" / "test_split.csv") if args.own_test_only else None
    index = index_clips(source_root, allowed)
    rows = []
    for plan_number, (left_kind, right_kind, audio_from) in enumerate(PLANS):
        pairs = pick_pairs(index, left_kind, right_kind, args.per_plan, args.seed + plan_number)
        for pair_number, (left, right) in enumerate(tqdm(pairs, desc=f"{left_kind}+{right_kind} audio {audio_from}")):
            sample_id = f"{left_kind}_{right_kind}_{audio_from}_{pair_number:03d}"
            out = raw_out / f"{sample_id}{'.mkv' if args.lossless else '.mp4'}"
            if not out.exists():
                tile_pair(Path(args.ffmpeg), left, right, out, audio_from, args.height, args.lossless)
            side = fake_side(left_kind, right_kind)
            rows.append({
                "sample_id": sample_id, "raw_path": str(out.relative_to(out_root)),
                "left_clip": str(left.relative_to(source_root)), "right_clip": str(right.relative_to(source_root)),
                "left_kind": left_kind, "right_kind": right_kind, "audio_from": audio_from,
                "fake_side": side, "label": int(side != "none"),
            })

    with open(labels_out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} tiled videos in {raw_out}")


if __name__ == "__main__":
    main()
