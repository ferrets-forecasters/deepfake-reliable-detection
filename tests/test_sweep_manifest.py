import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "compression_sweep.py"
spec = importlib.util.spec_from_file_location("compression_sweep", SCRIPT)
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def write(path, rows):
    path.parent.mkdir(parents=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep.MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_relative_clip_paths_are_read_from_the_dataset_folder(tmp_path):
    manifest = tmp_path / "sweep" / "labels" / "manifest.csv"
    write(manifest, [{"video_id": "a", "dataset": "hidf", "label": 1, "condition": "crf23", "path": "raw/crf23/a.mp4", "bytes": 10}])
    (row,) = sweep.read_manifest(manifest)
    assert row["path"] == tmp_path / "sweep" / "raw" / "crf23" / "a.mp4" and row["label"] == 1 and row["bytes"] == 10


def test_absolute_clip_paths_are_kept(tmp_path):
    absolute = tmp_path / "x.mp4"
    write(tmp_path / "s" / "labels" / "manifest.csv", [{"video_id": "a", "dataset": "hidf", "label": 0, "condition": "original", "path": str(absolute), "bytes": 1}])
    assert sweep.read_manifest(tmp_path / "s" / "labels" / "manifest.csv")[0]["path"] == absolute


def test_metrics_of_a_perfect_ranking():
    m = sweep.metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auc"] == 1.0 and m["real_flagged"] == 0.0 and m["fake_caught"] == 1.0
