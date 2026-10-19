from __future__ import annotations

import argparse
import csv

from tqdm import tqdm

from deepfake_detection.config import load_config

FIELDS = ["case_id", "query", "rank", "title", "uploader", "duration_s", "views", "url"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Metadata of YouTube search results for each query of datasets/viral_cases/labels/search_queries.csv, nothing is downloaded. Resumable: a finished query is skipped. Run with `uv run --with yt-dlp python scripts/find_viral_videos.py`.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results", type=int, default=8)
    args = parser.parse_args()

    import yt_dlp

    config = load_config(args.config)
    labels = config.paths.datasets_root / "viral_cases" / "labels"
    with open(labels / "search_queries.csv", encoding="utf-8", newline="") as f:
        queries = list(csv.DictReader(f))
    out_path = labels / "search_results.csv"
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8", newline="") as f:
            done = {(r["case_id"], r["query"]) for r in csv.DictReader(f)}
    new_file = not out_path.exists()
    with open(out_path, "a", encoding="utf-8", newline="") as f, yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True}) as ydl:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for q in tqdm([q for q in queries if (q["case_id"], q["query"]) not in done], desc="queries", unit="query"):
            try:
                entries = ydl.extract_info(f"ytsearch{args.results}:{q['query']}", download=False).get("entries", [])
            except Exception as error:
                print(f"{q['query']}: {type(error).__name__}: {error}")
                continue
            for rank, e in enumerate(entries, 1):
                writer.writerow({"case_id": q["case_id"], "query": q["query"], "rank": rank, "title": e.get("title"), "uploader": e.get("uploader") or e.get("channel"),
                                 "duration_s": e.get("duration"), "views": e.get("view_count"), "url": e.get("url")})
            f.flush()


if __name__ == "__main__":
    main()
