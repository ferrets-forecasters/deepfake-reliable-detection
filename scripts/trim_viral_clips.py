from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from deepfake_detection.config import load_config

FFMPEG = Path(r"C:\tools\ffmpeg\bin\ffmpeg.exe")
CASE_FIELDS = ["case_id", "title", "date", "platform", "source_url", "claim", "basis_url", "language", "license_note", "file", "note"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cuts the stretches listed in datasets/viral_cases/labels/trims.csv (file, start_s, end_s, claim, note) out of the downloaded news segments and adds each cut to labels/cases.csv with its claim. Resumable: an existing cut is skipped.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.paths.datasets_root / "viral_cases"
    with open(root / "labels" / "cases.csv", encoding="utf-8-sig", newline="") as f:
        cases = list(csv.DictReader(f))
    by_file = {c["file"]: c for c in cases}
    known = {c["case_id"] for c in cases}
    with open(root / "labels" / "trims.csv", encoding="utf-8", newline="") as f:
        trims = list(csv.DictReader(f))

    added = []
    for t in trims:
        source = by_file[t["file"]]
        start, end = float(t["start_s"]), float(t["end_s"])
        stem = f"{Path(t['file']).stem}_{int(start)}-{int(end)}"
        target = root / "raw" / f"{stem}.mp4"
        if not target.exists():
            subprocess.run([str(FFMPEG), "-y", "-loglevel", "error", "-ss", str(start), "-to", str(end), "-i", str(root / "raw" / t["file"]),
                            "-c:v", "libx264", "-crf", "18", "-c:a", "aac", str(target)], check=True)
        if stem not in known:
            added.append({**source, "case_id": stem, "title": f"{source['title']} [{start:g}-{end:g} s]", "claim": t["claim"], "file": target.name,
                          "note": f"cut of {t['file']}: {t['note']}"})
    if added:
        with open(root / "labels" / "cases.csv", "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CASE_FIELDS)
            writer.writerows(added)
    print(f"{len(trims)} cuts, {len(added)} added to cases.csv")


if __name__ == "__main__":
    main()
