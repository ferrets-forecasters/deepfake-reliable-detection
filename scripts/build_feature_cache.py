from __future__ import annotations

import argparse

from deepfake_detection.config import load_config
from deepfake_detection.config.config import _project_root
from deepfake_detection.training.reference_training import DATASET_ROOTS, _shared_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shared memory-mapped embedding cache of the five datasets once, so no training run has to read the 223,445 embedding files itself. Does nothing if it exists.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = sorted(str(f) for dataset in DATASET_ROOTS.values() for f in config.embedding_dir(dataset).rglob("*.npz"))
    cache = _shared_cache(paths, _project_root() / "temp" / "training_cache" / "features", args.workers)
    print(f"shared cache: {len(cache)} of {len(paths)} embedding files readable", flush=True)


if __name__ == "__main__":
    main()
