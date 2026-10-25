from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config

SEGMENT_SECONDS = 2.0


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def manifest_row(manifest: dict[str, dict], sample: str) -> dict | None:
    """The embedding manifest row of a sample; PolyGlotFake real videos are listed as real_<id> in the test lists and as <id> in the manifest."""
    return manifest.get(sample) or manifest.get(sample.removeprefix("real_"))


def probe(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
    first = capture.read()[0] if opened else False
    capture.release()
    seconds = frames / fps if fps > 0 else 0.0
    return {"opened": int(opened), "first_frame": int(first), "frames": int(frames), "fps": round(fps, 3), "seconds": round(seconds, 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the extracted test videos of celebdfv2, ffpp and polyglotfake: each file opens and decodes its first frame, its label agrees with the "
                                                 "embedding manifest, and its length gives the same number of 2-second segments as the embeddings of the same sample "
                                                 "(a wrong video would rarely match). Writes verification.csv and verification.yaml next to the inventory.")
    parser.add_argument("--datasets", nargs="*", default=["celebdfv2", "ffpp", "polyglotfake"])
    args = parser.parse_args()
    config = load_config()
    out_dir = config.paths.analysis_dir("test_video_inventory")
    fetch = {(r["dataset"], r["archive_member"]): r for r in read_csv(out_dir / "extracted_videos.csv")}
    rows, summary = [], {}
    for dataset in args.datasets:
        manifest = {r["sample_id"]: r for r in read_csv(config.labels_dir(dataset) / "manifest.csv")}
        listed = read_csv(out_dir / f"{dataset}.csv") + (read_csv(out_dir / f"{dataset}_official.csv") if (out_dir / f"{dataset}_official.csv").exists() else [])
        inventory = list({r["archive_member"]: r for r in listed if (dataset, r["archive_member"]) in fetch}.values())
        counts = {"files": len(inventory), "missing_file": 0, "unreadable": 0, "no_embedding": 0, "segments_equal": 0, "segments_off_by_one": 0, "segments_differ": 0, "label_differs": 0}
        for item in tqdm(inventory, desc=dataset, unit="video"):
            path = config.raw_dir(dataset) / "test_split" / item["archive_member"]
            row = {"dataset": dataset, "sample_id": item["sample_id"], "label": item["label"], "path": str(path.relative_to(config.raw_dir(dataset))), "status": ""}
            if not path.exists():
                counts["missing_file"] += 1
                rows.append({**row, "status": "missing file"})
                continue
            info = probe(path)
            row.update(info)
            if not (info["opened"] and info["first_frame"] and info["frames"] > 0):
                counts["unreadable"] += 1
                rows.append({**row, "status": "unreadable"})
                continue
            expected = manifest_row(manifest, item["sample_id"])
            if expected is None:
                counts["no_embedding"] += 1
                rows.append({**row, "status": "ok, no embedding to compare"})
                continue
            segments = math.floor(info["seconds"] / SEGMENT_SECONDS + 1e-6)
            row.update({"segments_video": segments, "segments_embedding": int(expected["num_segments"])})
            gap = abs(segments - int(expected["num_segments"]))
            counts["segments_equal" if gap == 0 else "segments_off_by_one" if gap == 1 else "segments_differ"] += 1
            if expected["label"] != item["label"]:
                counts["label_differs"] += 1
            rows.append({**row, "status": "ok" if gap <= 1 and expected["label"] == item["label"] else "check"})
        summary[dataset] = counts
    columns = ["dataset", "sample_id", "label", "path", "status", "opened", "first_frame", "frames", "fps", "seconds", "segments_video", "segments_embedding"]
    with open(out_dir / "verification.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "verification.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))
    print("rows to look at:", sum(r["status"] in ("check", "unreadable", "missing file") for r in rows))


if __name__ == "__main__":
    main()
