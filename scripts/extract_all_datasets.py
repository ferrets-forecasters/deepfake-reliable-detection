"""
Run feature extraction for all datasets sequentially.
Already-extracted segments are skipped automatically (resumable).

Usage:
    uv run python scripts/extract_all_datasets.py

Optional flags passed through to extract_embeddings.py:
    --ffmpeg-timeout 120
    --probe-timeout  30
"""
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT  = Path(__file__).with_name("extract_embeddings.py")

DATASETS = [
    "polyglotfake",
    "hidf",
    "celebdfv2",
    "fakeavceleb",
    "ffpp",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg-timeout", type=int, default=120)
    parser.add_argument("--probe-timeout",  type=int, default=30)
    parser.add_argument("--only", nargs="+", choices=DATASETS, default=None,
                        help="Run only these datasets")
    parser.add_argument("--skip", nargs="+", choices=DATASETS, default=None,
                        help="Skip these datasets")
    args = parser.parse_args()

    targets = args.only if args.only else DATASETS
    if args.skip:
        targets = [d for d in targets if d not in args.skip]

    print("=" * 60)
    print(f"  Feature Extraction - {len(targets)} dataset(s)")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    for d in targets:
        print(f"  - {d}")
    print()

    results = {}

    for dataset in targets:
        print(f"\n{'-'*60}")
        print(f"  [{targets.index(dataset)+1}/{len(targets)}]  {dataset.upper()}")
        print(f"{'-'*60}")

        cmd = [
            sys.executable, str(SCRIPT),
            "--dataset", dataset,
            "--ffmpeg-timeout", str(args.ffmpeg_timeout),
            "--probe-timeout",  str(args.probe_timeout),
        ]

        start = datetime.now()
        ret   = subprocess.run(cmd)
        elapsed = datetime.now() - start
        mins, secs = divmod(int(elapsed.total_seconds()), 60)

        status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
        results[dataset] = (status, f"{mins}m {secs}s")
        print(f"\n  {dataset}: {status}  [{mins}m {secs}s]")

    # -- Final summary ---------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DONE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    for dataset, (status, elapsed) in results.items():
        ok = "[ok]" if status == "OK" else "[failed]"
        print(f"  {ok} {dataset:<16}  {status:<20}  {elapsed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
