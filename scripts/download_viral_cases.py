from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from deepfake_detection.config import load_config
from deepfake_detection.utils.tools import find_tool


def main() -> None:
    parser = argparse.ArgumentParser(description="Downloads the videos listed in datasets/viral_cases/labels/downloads.csv with yt-dlp into datasets/viral_cases/raw/. --probe prints the size of each file and downloads nothing. Resumable: an existing file is skipped. Run with `uv run --with yt-dlp python scripts/download_viral_cases.py`.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--list", type=Path, default=None)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--max-mb", type=float, default=300, help="skip a file whose estimated size is larger")
    args = parser.parse_args()

    import yt_dlp

    config = load_config(args.config)
    root = config.paths.datasets_root / "viral_cases"
    with open(args.list or root / "labels" / "downloads.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    fmt = f"bv*[height<={args.max_height}]+ba/b[height<={args.max_height}]/b"
    log = open(root / "labels" / "download.log", "a", encoding="utf-8")
    for row in rows:
        target = raw / row["file"]
        if target.exists():
            print(f"{row['file']}: already there ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        base = {"format": fmt, "outtmpl": str(target.with_suffix(".%(ext)s")), "merge_output_format": "mp4", "quiet": True, "no_warnings": True,
                "ffmpeg_location": str(find_tool("ffmpeg").parent), "noprogress": True, "retries": 5, "fragment_retries": 5}
        attempts = [{}, {"extractor_args": {"youtube": {"player_client": ["tv"]}}}, {"extractor_args": {"youtube": {"player_client": ["android"]}}},
                    {"format": f"b[height<={args.max_height}]/b"}]
        size, line = 0, row["file"]
        for number, extra in enumerate(attempts, 1):
            try:
                with yt_dlp.YoutubeDL({**base, **extra}) as ydl:
                    info = ydl.extract_info(row["source_url"], download=False)
                    chosen = info.get("requested_formats") or [info]
                    size = sum((c.get("filesize") or c.get("filesize_approx") or 0) for c in chosen)
                    height = max((c.get("height") or 0) for c in chosen)
                    line = f"{row['file']}: {info.get('title', '')[:70]!r} by {info.get('uploader')}, {info.get('duration')} s, {height}p, about {size / 1e6:.1f} MB"
                    print(line.encode("ascii", "replace").decode(), flush=True)
                    if args.probe:
                        break
                    if size / 1e6 > args.max_mb:
                        print(f"  skipped: above {args.max_mb} MB")
                        break
                    ydl.download([row["source_url"]])
                    log.write(f"{time.strftime('%F %T')} {line} (attempt {number})" + chr(10))
                    log.flush()
                    break
            except yt_dlp.utils.DownloadError as error:
                message = f"  attempt {number} failed: {str(error)[:120]}"
                print(message, flush=True)
                log.write(f"{time.strftime('%F %T')} {row['file']} {message.strip()}" + chr(10))
                log.flush()
                time.sleep(5)


if __name__ == "__main__":
    main()
