from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

from deepfake_detection.config import load_config

SCRIPTS = Path(__file__).parent
PAIR = {"real": "RealVideo-RealAudio/African/men/id00076/00109.mp4", "fake": "FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.mp4"}


def videos(folder: Path, *suffixes: str) -> list[str]:
    return sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in suffixes)


def steps(config, model: str) -> list[tuple[str, list[str]]]:
    root = config.paths.datasets_root
    fakeavceleb = config.raw_dir("fakeavceleb")
    manifest = list(csv.DictReader(open(root / "fakeavceleb_reencoded" / "labels" / "manifest.csv", encoding="utf-8")))
    originals = [str(next(fakeavceleb.rglob(row["source_clip"].replace("\\", "/")))) for row in manifest]
    pair = {key: next(fakeavceleb.rglob(value)) for key, value in PAIR.items()}
    m = ["--model", model]
    seed_tests = [(f"expert_tests_seed{seed}", ["expert_tests.py", "--model", f"moe_combined_seed{seed}", "--split-seed", str(seed)]) for seed in (43, 44)]
    summaries = [
        ("summary_conversation_and_control", ["summarize_multi_subject.py", *m, "--conversation", "fakeavceleb_conversation_lossless", "--control", "fakeavceleb_reencoded", "--original", "fakeavceleb_original"]),
        ("summary_tiled_first_segment", ["summarize_multi_subject.py", *m, "--tiled", "fakeavceleb_tiled_lossless"]),
        ("summary_tiled_every_segment", ["summarize_multi_subject.py", *m, "--conversation", "fakeavceleb_tiled_lossless"]),
    ]
    return [
        *seed_tests,
        ("embedding_analyses", ["analyze_embeddings.py"]),
        ("conversation_lossless", ["demo.py", "--video", *videos(root / "fakeavceleb_conversation_lossless" / "raw", ".mkv"), "--name", "fakeavceleb_conversation_lossless", *m]),
        ("tiled_lossless", ["demo.py", "--video", *videos(root / "fakeavceleb_tiled_lossless" / "raw", ".mkv"), "--name", "fakeavceleb_tiled_lossless", *m]),
        ("tiled_first_segment", ["score_tiled_set.py", "--dataset", "fakeavceleb_tiled_lossless", *m]),
        ("reencoded", ["infer_multi_subject.py", str(root / "fakeavceleb_reencoded" / "raw"), "--name", "fakeavceleb_reencoded", *m]),
        ("original", ["infer_multi_subject.py", *originals, "--name", "fakeavceleb_original", *m]),
        *summaries,
        ("still_frames", ["score_still_frames.py", *m]),
        ("explain_pair_real", ["explain_clip.py", str(pair["real"]), *m]),
        ("explain_pair", ["explain_clip.py", str(pair["fake"]), "--other-video", str(pair["real"]), *m]),
        ("openforensics", ["score_images.py", *m]),
        ("speed_benchmark", ["benchmark_inference.py", "--models", model, "concat_combined_retrain"]),
        ("report", ["analyze_training.py", "--skip-export"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run the inference experiments of the paper with another model: two-person sets, re-encoding control, OpenForensics, still frames, the explanation pair. Resumable: a finished step is listed in results/metrics/rerun_inference_<model>/done.txt and skipped.")
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out = config.paths.results_dir(f"rerun_inference_{args.model}")
    out.mkdir(parents=True, exist_ok=True)
    done_file = out / "done.txt"
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.exists() else set()
    plan = [s for s in steps(config, args.model) if not args.only or s[0] in args.only]
    for number, (name, command) in enumerate(plan, 1):
        if name in done:
            print(f"[{number}/{len(plan)}] {name}: done before", flush=True)
            continue
        started = time.time()
        print(f"[{number}/{len(plan)}] {name}: start {time.strftime('%H:%M:%S')}", flush=True)
        with open(out / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.run([sys.executable, str(SCRIPTS / command[0]), *command[1:]], stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"[{number}/{len(plan)}] {name}: exit {code} after {time.time() - started:.0f} s", flush=True)
        if code == 0:
            with open(done_file, "a", encoding="utf-8") as f:
                f.write(name + "\n")


if __name__ == "__main__":
    main()
