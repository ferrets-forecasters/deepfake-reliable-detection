from __future__ import annotations

import argparse
import csv
from pathlib import Path

from deepfake_detection.config import Config, load_config


def _local_status(raw_dir: Path) -> tuple[str, str]:
    if not raw_dir.is_dir():
        return "not started", ""
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file())
    if not files:
        return "not started", ""
    total_bytes = sum(p.stat().st_size for p in files)
    total_mb = total_bytes / (1024 * 1024)
    listing = "; ".join(f"{p.name} ({p.stat().st_size / (1024 * 1024):.1f} MB)" for p in files)
    return f"{len(files)} file(s), {total_mb:.1f} MB", listing


def _rows(config: Config) -> list[dict[str, str]]:
    rows = []
    for spec in config.datasets:
        raw_dir = config.paths.raw_dir / spec.name
        status, listing = _local_status(raw_dir)
        rows.append(
            {
                "name": spec.name,
                "modality": spec.modality,
                "real": spec.real,
                "fake": spec.fake,
                "segments": spec.segments,
                "access": spec.access,
                "source_url": spec.source_url,
                "download_url": spec.download_url,
                "local_path": str(raw_dir),
                "local_status": status,
                "local_files": listing,
            }
        )
    return rows


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Manifest",
        "",
        "Registry of the five datasets the replication pipeline trains on: expected segment counts from the paper, "
        "access route, and current local download status. Regenerate with `scripts/export_dataset_registry.py` "
        "after downloading or requesting access to a dataset; source of truth is `configs/default.yaml`.",
        "",
        "| Dataset | Modality | Real | Fake | Segments | Access | Local status |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['modality']} | {row['real']} | {row['fake']} | {row['segments']} | "
            f"{row['access']} | {row['local_status']} |"
        )

    lines += ["", "## Access links", ""]
    for row in rows:
        lines.append(f"- **{row['name']}**: source {row['source_url']}, download/request {row['download_url']}")

    lines += ["", "## Local files", ""]
    for row in rows:
        lines.append(f"- **{row['name']}** (`{row['local_path']}`): {row['local_files'] or '(none yet)'}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate datasets/manifest.csv and datasets/manifest.md.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    for spec in config.datasets:
        (config.paths.raw_dir / spec.name).mkdir(parents=True, exist_ok=True)

    rows = _rows(config)
    _write_csv(rows, config.paths.datasets_root / "manifest.csv")
    _write_markdown(rows, config.paths.datasets_root / "manifest.md")
    print(f"wrote {config.paths.datasets_root / 'manifest.csv'} and {config.paths.datasets_root / 'manifest.md'}")


if __name__ == "__main__":
    main()
