from __future__ import annotations

import argparse

from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.embedding_manifest import build_embedding_manifest, manifest_totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Write datasets/<name>/labels/manifest.csv from the embedding folders.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="+", default=None, help="default: every dataset in datasets.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    for name in tqdm(args.datasets or config.dataset_names(), desc="datasets"):
        spec = config.dataset(name)
        manifest = build_embedding_manifest(spec, config.embedding_dir(name), config.manifest_columns)
        path = config.manifest_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False, encoding="utf-8")
        totals = manifest_totals(manifest)
        expected = {"real": spec.real, "fake": spec.fake, "segments": spec.segments}
        status = "match" if totals == expected else f"MISMATCH expected {expected}"
        tqdm.write(f"{name}: {len(manifest)} samples, segments {totals}, {status} -> {path}")


if __name__ == "__main__":
    main()
