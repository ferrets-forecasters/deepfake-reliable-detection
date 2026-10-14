from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml

from deepfake_detection.config import load_config

DEFAULT_SETS = ["moe_combined_retrain_on_fakeavceleb_tiled_own_test_lossless"]  # the only tiled set scored with the speaker diarizer; the older sets used one shared speaker, so their count is 0 by construction
BLOCK = 40


def speakers_in(path: Path) -> tuple[int, int]:
    turns = json.loads(path.read_text(encoding="utf-8")).get("turns", [])
    return len({t["speaker"] for t in turns}), len(turns)


def summarize(rows: list[dict]) -> dict:
    split = [r for r in rows if r["speakers"] >= 2]
    by_plan = {plan: {"videos": n, "split_into_two_or_more": sum(1 for r in split if r["plan"] == plan)} for plan, n in sorted(Counter(r["plan"] for r in rows).items())}
    blocks = {f"{i + 1}-{i + len(chunk)}": sum(1 for r in chunk if r["speakers"] >= 2) for i in range(0, len(rows), BLOCK) for chunk in [rows[i:i + BLOCK]]}
    return {"videos": len(rows), "split_into_two_or_more": len(split), "share": round(len(split) / len(rows), 4) if rows else None,
            "by_plan": by_plan, f"by_block_of_{BLOCK}_in_sorted_order": blocks}


def main() -> None:
    parser = argparse.ArgumentParser(description="How often one voice is split into two speakers: the speaker count in the saved turns of every video of the single-audio-track tiled sets, by plan and by block of 40, and the agreement with the 40-video export in diarization_matrices. Reads the saved JSON only.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sets", nargs="+", default=DEFAULT_SETS, help="folders under results/inferences with one JSON per video")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.analysis_dir("single_voice_split_count")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {}
    per_video: list[dict] = []
    for name in args.sets:
        folder = config.paths.predictions_dir(name)
        files = sorted(folder.glob("*.json"))
        if not files:
            report[name] = "no JSON files"
            continue
        rows = []
        for path in files:
            speakers, turns = speakers_in(path)
            rows.append({"set": name, "video": path.stem, "plan": re.sub(r"_\d+$", "", path.stem), "speakers": speakers, "turns": turns})
        report[name] = summarize(rows)
        per_video += rows

    manifest = config.paths.analysis_dir("diarization_matrices") / "manifest.csv"
    if manifest.exists():
        with open(manifest, encoding="utf-8", newline="") as f:
            export = {Path(r["video"].replace("\\", "/")).stem: int(r["speakers_found"]) for r in csv.DictReader(f) if r["set"] == "single_voice"}
        saved = {r["video"]: r["speakers"] for r in per_video if r["set"] == DEFAULT_SETS[0] or r["set"] == args.sets[0]}
        common = sorted(set(export) & set(saved))
        report["agreement_with_diarization_matrices_export"] = {
            "export_videos": len(export), "also_in_saved_turns": len(common),
            "same_speaker_count": sum(1 for v in common if export[v] == saved[v]),
            "export_split": sum(1 for v in export.values() if v >= 2), "saved_turns_split_on_the_same_videos": sum(1 for v in common if saved[v] >= 2),
        }

    (out_dir / "summary.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    with open(out_dir / "per_video.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "video", "plan", "speakers", "turns"])
        writer.writeheader()
        writer.writerows(per_video)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
