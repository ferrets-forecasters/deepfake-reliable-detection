from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait until a progress file stops changing and print its last line. Exits when the file has not been written for --stall-minutes, or when it ends with 100%.")
    parser.add_argument("progress_file", type=Path)
    parser.add_argument("--stall-minutes", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args()

    started = time.time()
    while True:
        time.sleep(args.poll_seconds)
        if not args.progress_file.exists():
            if time.time() - started > args.stall_minutes * 60:
                print(f"no progress file after {args.stall_minutes:.0f} minutes: {args.progress_file}")
                return
            continue
        lines = args.progress_file.read_text(encoding="utf-8").strip().splitlines()
        last = lines[-1] if lines else ""
        age = (time.time() - args.progress_file.stat().st_mtime) / 60
        if age > args.stall_minutes:
            print(f"STALLED for {age:.0f} minutes. Last line: {last}")
            return
        if "100%" in last:
            print(f"DONE. Last line: {last}")
            return


if __name__ == "__main__":
    main()
