from __future__ import annotations

import argparse
import csv
import logging
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import _project_root

FETCH_LIST = "extracted_videos.csv"


def zip_entries(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    return [info for info in archive.infolist() if not info.is_dir()]


def inner_archives(outer: zipfile.ZipFile):
    """The per-dataset zips inside the outer zip, opened in place (stored entries, so nothing is unpacked to disk first): (dataset, ZipFile)."""
    for info in outer.infolist():
        if info.filename.lower().endswith(".zip"):
            yield Path(info.filename).stem, zipfile.ZipFile(outer.open(info))


def match_rows(rows: list[dict], entries: list[zipfile.ZipInfo]) -> tuple[dict[int, zipfile.ZipInfo], dict[str, list[int]]]:
    """Pair each row of fetch_list.csv with a zip entry: the entry path ends with the member path (exact), else the one entry with the same file name (name).
    Returns row index -> entry, and the indexes by outcome: exact, name, ambiguous, missing."""
    by_name: dict[str, list[zipfile.ZipInfo]] = {}
    for info in entries:
        by_name.setdefault(PurePosixPath(info.filename).name, []).append(info)
    paired, outcome = {}, {"exact": [], "name": [], "ambiguous": [], "missing": []}
    for index, row in enumerate(rows):
        member = row["archive_member"]
        candidates = by_name.get(PurePosixPath(member).name, [])
        exact = [c for c in candidates if c.filename == member or c.filename.endswith("/" + member)]
        if len(exact) == 1:
            paired[index] = exact[0]
            outcome["exact"].append(index)
        elif not exact and len(candidates) == 1:
            paired[index] = candidates[0]
            outcome["name"].append(index)
        else:
            outcome["ambiguous" if candidates else "missing"].append(index)
    return paired, outcome


def target_path(config, row: dict) -> Path:
    return config.raw_dir(row["dataset"]) / "test_split" / row["archive_member"]


def report(dataset, rows, entries, paired, outcome) -> None:
    total = sum(info.file_size for info in entries)
    kinds = Counter(PurePosixPath(info.filename).suffix.lower() for info in entries)
    wanted = {id(info) for info in paired.values()}
    size = sum(info.file_size for info in paired.values())
    print(f"{dataset}: {len(entries)} entries {dict(kinds)}, {total / 1e9:.2f} GB; list rows {len(rows)}: "
          + ", ".join(f"{key} {len(value)}" for key, value in outcome.items())
          + f"; paired {len(paired)} ({size / 1e9:.2f} GB); entries the list does not name {sum(1 for info in entries if id(info) not in wanted)}")


def extract(config, rows, paired, archive: zipfile.ZipFile, log: logging.Logger, name: str) -> tuple[int, int]:
    """Write every paired entry to datasets/<dataset>/raw/test_split/<member>. zipfile checks the CRC-32 of each entry while it reads it. A finished file
    of the right size is skipped, so a rerun continues where it stopped; a file is written under a .part name and renamed when complete."""
    done = skipped = 0
    started, written = time.time(), 0
    for index in tqdm(sorted(paired), desc=f"extract {name}", unit="video"):
        info, row = paired[index], rows[index]
        path = target_path(config, row)
        if path.exists() and path.stat().st_size == info.file_size:
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        with archive.open(info) as source, open(part, "wb") as sink:
            while chunk := source.read(1 << 22):
                sink.write(chunk)
        part.replace(path)
        done += 1
        written += info.file_size
        if done % 100 == 0:
            log.info("%s: %d written, %d skipped, %.2f GB, %.1f MB/s", name, done, skipped, written / 1e9, written / 1e6 / max(time.time() - started, 1e-6))
    log.info("%s: finished, %d written, %d skipped", name, done, skipped)
    return done, skipped


def confirm(config, rows, paired) -> int:
    """Every paired entry has its file on disk with the size the zip states. Returns the number that does not."""
    bad = 0
    for index, info in paired.items():
        path = target_path(config, rows[index])
        if not path.exists() or path.stat().st_size != info.file_size:
            bad += 1
            print("missing or wrong size:", path)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Read the test-video zip that Fathan cut from the shared Drive archives (an outer zip with one zip per dataset and a report.txt), "
                                                 "match each inner zip with fetch_list.csv, and optionally extract the videos to datasets/<dataset>/raw/test_split/ with a "
                                                 "progress bar, a log, a CRC check while reading and a size check of every file.")
    parser.add_argument("zip", type=Path)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--list", type=Path, help="defaults to extracted_videos.csv of results/analyses/test_video_inventory")
    args = parser.parse_args()
    config = load_config()
    list_path = args.list or config.paths.analysis_dir("test_video_inventory") / FETCH_LIST
    with open(list_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    if args.extract:
        log_dir = _project_root() / "temp" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=log_dir / "extract_test_video_zip.log", level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")
    log, wrong = logging.getLogger("extract"), 0
    with zipfile.ZipFile(args.zip) as outer:
        for info in outer.infolist():
            if info.filename.lower().endswith(".txt"):
                print(f"--- {info.filename}\n{outer.read(info).decode('utf-8', 'replace')}")
        for dataset, archive in inner_archives(outer):
            rows = [r for r in all_rows if r["dataset"] == dataset]
            entries = zip_entries(archive)
            paired, outcome = match_rows(rows, entries)
            report(dataset, rows, entries, paired, outcome)
            if args.extract:
                done, skipped = extract(config, rows, paired, archive, log, dataset)
                bad = confirm(config, rows, paired)
                wrong += bad
                print(f"{dataset}: extracted {done}, already on disk {skipped}, missing or wrong size {bad}")
    if args.extract:
        print(f"total files missing or of the wrong size: {wrong}")


if __name__ == "__main__":
    main()
