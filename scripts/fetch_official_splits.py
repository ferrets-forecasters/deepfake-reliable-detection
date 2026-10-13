from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

from deepfake_detection.config import load_config

SPLIT_FILES = {
    "ffpp": [
        ("train.json", "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json"),
        ("val.json", "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json"),
        ("test.json", "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json"),
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the split files the dataset authors publish into datasets/<name>/labels/official_split/. Skips files already present.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=list(SPLIT_FILES))
    args = parser.parse_args()

    config = load_config(args.config)
    for dataset in args.datasets:
        folder = config.paths.datasets_root / dataset / "labels" / "official_split"
        folder.mkdir(parents=True, exist_ok=True)
        for name, url in SPLIT_FILES[dataset]:
            target = folder / name
            if target.exists():
                print(f"{dataset}/{name}: present")
                continue
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            target.write_bytes(data)
            print(f"{dataset}/{name}: {len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
