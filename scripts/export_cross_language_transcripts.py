from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from deepfake_detection.config import load_config

ARCHIVE_URL = "https://drive.usercontent.google.com/download?id=1aBWLii-TbrpKNLSTwpmjqu98eKovWLxF&export=download&confirm=t"
FAKE_NAME = re.compile(r"^fake_(?P<src>[a-z]{2})_(?P<n>\d+)_to_(?P<tgt>[a-z]{2})_(?P<voice>.+)$")
NO_PARTITION = "not in the embedded data"
SOURCE_NOTE = "automatic transcript by Whisper medium, the dataset supplies no text"


def choose_pairs(config, count: int, seed: int) -> list[dict]:
    """One test-split fake per source real clip, drawn with a seeded shuffle. Real clips that sit in the test split come first, then real
    clips that are in no partition (never used in training); a source and target language pair is used at most once."""
    path = config.paths.analysis_dir("polyglotfake_split_check") / "pairs.csv"
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["fake_partition"] == "test" and r["source_partition"] in {"test", NO_PARTITION}]
    random.Random(seed).shuffle(rows)
    rows.sort(key=lambda r: r["source_partition"] != "test")
    chosen, seen, reals = [], set(), set()
    for r in rows:
        m = FAKE_NAME.match(r["fake"])
        if not m or (m["src"], m["tgt"]) in seen or r["source_real"] in reals:
            continue
        seen.add((m["src"], m["tgt"]))
        reals.add(r["source_real"])
        chosen.append({"src": m["src"], "tgt": m["tgt"], "n": m["n"], "voice": m["voice"], "fake": r["fake"], "real": r["source_real"],
                       "real_partition": "test" if r["source_partition"] == "test" else "none"})
        if len(chosen) == count:
            break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcripts of cross-language PolyGlotFake pairs in the original script: the real clip and its fake in another language, both from the seed-42 test split. Members are fetched from the archive by ranged requests.")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    pairs = choose_pairs(config, args.count, args.seed)
    raw = config.paths.datasets_root / "polyglotfake" / "raw" / "test_pairs"
    members = []
    for p in pairs:
        members += [f"PolyGlotFake/real/{p['src']}/{p['src']}_{p['n']}.mp4", f"PolyGlotFake/fake/to_{p['tgt']}/{p['src']}_{p['n']}_to_{p['tgt']}_{p['voice']}.mp4"]
    subprocess.run([sys.executable, str(Path(__file__).parent / "remote_rar.py"), "--url", ARCHIVE_URL, "--out", str(raw), "--member", *members], check=False)

    import torch
    import whisper

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_memory_fraction and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    model = whisper.load_model(args.model, device=device)
    out_dir = config.paths.analysis_dir("cross_language_transcripts")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, p in enumerate(tqdm(pairs, desc="pairs")):
        for role, member, language in (("real", members[2 * number], p["src"]), ("fake", members[2 * number + 1], p["tgt"])):
            path = raw / member
            if not path.exists():
                continue
            result = model.transcribe(str(path), language=language, fp16=device == "cuda", condition_on_previous_text=False)
            rows.append({"pair": f"pair_{number:02d}", "role": role, "clip": Path(member).stem, "source_language": p["src"], "target_language": p["tgt"],
                         "transcribed_as": language, "text_original_script": result["text"].strip(), "source": SOURCE_NOTE,
                         "fake_partition": "test", "real_partition": p["real_partition"], "path": str(path),
                         "why_chosen": "one test fake per real clip, test-split reals first then reals in no partition, seeded draw"})
    with open(out_dir / "transcripts.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} transcripts of {len(pairs)} pairs in {out_dir}")


if __name__ == "__main__":
    main()
