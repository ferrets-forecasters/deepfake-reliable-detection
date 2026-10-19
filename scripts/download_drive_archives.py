from __future__ import annotations

import argparse

import gdown

from deepfake_detection.config import load_config
from deepfake_detection.data.drive_archives import load_drive_archives



def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset archives from the shared Drive into datasets/<name>/raw/. Run with `uv run --with gdown python scripts/download_drive_archives.py`. A partial download resumes, a finished one is skipped.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=["celebdfv2", "polyglotfake"])
    args = parser.parse_args()

    config = load_config(args.config)
    archives = load_drive_archives()
    for dataset in args.datasets:
        file_id, name, expected = archives[dataset][0]
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
