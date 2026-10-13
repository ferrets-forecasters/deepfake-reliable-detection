from __future__ import annotations

import argparse
from pathlib import Path

import gdown

from deepfake_detection.config import load_config

# File IDs in the shared Drive folder 1UZ61KBsorHL0WdaLaCklDEG2-B17oenl (wiki/runpod-and-google-drive.md).
ARCHIVES = {
    "celebdfv2": ("1ycE03-UpDrrTKfem5FSewMNf5mtHL5Cw", "Celeb-DF-v2.zip", 9_952_957_051),
    "polyglotfake": ("1xH5DTDHghOsP3R8pQLUuSz_MfKY60yc5", "PolyGlotFake.rar", None),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset archives from the shared Drive into datasets/<name>/raw/. Run with `uv run --with gdown python scripts/download_drive_archives.py`. A partial download resumes, a finished one is skipped.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=list(ARCHIVES))
    args = parser.parse_args()

    config = load_config(args.config)
    for dataset in args.datasets:
        file_id, name, expected = ARCHIVES[dataset]
        target = config.raw_dir(dataset) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if expected and target.exists() and target.stat().st_size == expected:
            print(f"{dataset}: {name} present", flush=True)
            continue
        print(f"{dataset}: downloading {name}", flush=True)
        gdown.download(id=file_id, output=str(target), quiet=False, resume=True)
        size = target.stat().st_size
        print(f"{dataset}: {name} {size} bytes" + (" (matches)" if expected == size else f" (expected {expected})" if expected else ""), flush=True)


if __name__ == "__main__":
    main()
