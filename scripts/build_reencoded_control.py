from __future__ import annotations

import argparse
import csv
import subprocess

from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.tiling import ffmpeg_reencode_command
from deepfake_detection.utils.tools import find_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-encode the clips behind a tiled set one at a time, so a single face goes through the same encoder settings as in the tiled videos.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default="fakeavceleb_tiled")
    parser.add_argument("--name", default="fakeavceleb_reencoded")
    parser.add_argument("--per-kind", type=int, default=20)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--ffmpeg", default=None, help="ffmpeg executable; default: ffmpeg on PATH")
    args = parser.parse_args()

    config = load_config(args.config)
    source_root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    out_root = config.paths.datasets_root / args.name
    (out_root / "raw").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    with open(config.paths.datasets_root / args.dataset / "labels" / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clips: dict[str, list[str]] = {"real": [], "fake_av": []}
    for row in rows:
        for side in ("left", "right"):
            kind, clip = row[f"{side}_kind"], row[f"{side}_clip"]
            if kind in clips and clip not in clips[kind] and len(clips[kind]) < args.per_kind:
                clips[kind].append(clip)

    manifest = []
    for kind, names in clips.items():
        for number, clip in enumerate(tqdm(names, desc=kind)):
            out = out_root / "raw" / f"{kind}_{number:03d}.mp4"
            if not out.exists():
                subprocess.run(ffmpeg_reencode_command(find_tool("ffmpeg", args.ffmpeg), source_root / clip, out, args.height), check=True)
            manifest.append({"sample_id": out.stem, "raw_path": str(out.relative_to(out_root)), "source_clip": clip, "kind": kind, "label": int(kind != "real")})
    with open(out_root / "labels" / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "raw_path", "source_clip", "kind", "label"])
        writer.writeheader()
        writer.writerows(manifest)


if __name__ == "__main__":
    main()
