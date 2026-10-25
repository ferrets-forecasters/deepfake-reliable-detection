from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from list_test_videos import VIDEO_SUFFIXES, disk_index, official_rows, own_test_rows

FFPROBE = Path("C:/tools/ffmpeg/bin/ffprobe.exe")
NEW_DATASETS = ("celebdfv2", "ffpp", "polyglotfake")


def folder_label(dataset: str, path: Path) -> int:
    """Real (0) or fake (1) from the folder the file sits in, which is independent of every list and manifest."""
    parts = {p.lower() for p in path.parts}
    if dataset == "hidf":
        return 0 if "real-vid" in parts else 1
    if dataset == "fakeavceleb":
        return 0 if "realvideo-realaudio" in parts else 1
    return 0 if parts & {"real", "original", "original_sequences", "celeb-real", "youtube-real"} else 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 23):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> tuple[int, int, float, float]:
    capture = cv2.VideoCapture(str(path))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    return width, height, fps, frames / fps if fps else 0.0


def has_audio(path: Path) -> bool:
    out = subprocess.run([str(FFPROBE), "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(out.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent check of the test videos on disk: every video the own seed-42 test split and the official FF++ and Celeb-DF v2 test lists name is there, no file "
                                                 "outside those lists, the folder agrees with the label, no two files hold the same bytes, and the picture size, frame rate, length and audio "
                                                 "match what the dataset is. Writes set_check.yaml next to the inventory.")
    parser.add_argument("--audio-sample", type=int, default=100, help="files per dataset and label probed for an audio stream")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config()
    out_dir = config.paths.analysis_dir("test_video_inventory")
    own = own_test_rows(config)
    report: dict[str, dict] = {}
    hashes: dict[str, list[str]] = defaultdict(list)
    for dataset in config.dataset_names():
        raw = config.raw_dir(dataset)
        on_disk = disk_index(dataset, raw)
        wanted_own = {r["sample_id"]: int(r["label"]) for r in own if r["dataset"] == dataset}
        wanted_official = {o["sample_id"]: int(o["label"]) for o in official_rows(config, dataset)}
        wanted = {**wanted_official, **wanted_own}
        ids = [r["sample_id"] for r in own if r["dataset"] == dataset]
        entry = {"own_test_rows": len(ids), "sample_ids_shared_by_two_videos": len(ids) - len(set(ids)), "own_test": len(wanted_own), "official_test": len(wanted_official), "required": len(wanted),
                 "required_real": sum(v == 0 for v in wanted.values()), "required_fake": sum(v == 1 for v in wanted.values()),
                 "missing": sorted(set(wanted) - set(on_disk))[:10], "missing_count": len(set(wanted) - set(on_disk))}
        used = {k: on_disk[k] for k in wanted if k in on_disk}
        entry["label_differs_from_folder"] = sum(folder_label(dataset, p) != wanted[k] for k, p in used.items())
        if dataset in NEW_DATASETS:
            test_split = [p for p in (raw / "test_split").rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES]
            entry["files_in_test_split"] = len(test_split)
            entry["extra_files_not_in_any_list"] = sum(disk_key_of(dataset, p) not in wanted for p in test_split)
            with ThreadPoolExecutor(args.workers) as pool:
                for path, digest in zip(test_split, tqdm(pool.map(sha256, test_split), total=len(test_split), desc=f"sha256 {dataset}", unit="video")):
                    hashes[digest].append(f"{dataset}/{path.relative_to(raw).as_posix()}")
        groups: dict[int, list[Path]] = defaultdict(list)
        for k, p in used.items():
            groups[wanted[k]].append(p)
        stats = {}
        for label, paths in groups.items():
            sample = random.Random(42).sample(paths, min(len(paths), 300))
            info = [probe(p) for p in tqdm(sample, desc=f"probe {dataset} {'fake' if label else 'real'}", unit="video")]
            audio_sample = random.Random(42).sample(paths, min(len(paths), args.audio_sample))
            audio = sum(has_audio(p) for p in tqdm(audio_sample, desc="audio", unit="video"))
            stats["fake" if label else "real"] = {
                "videos": len(paths), "probed": len(info), "mb_mean": round(statistics.mean(p.stat().st_size for p in paths) / 1e6, 2),
                "width_x_height": dict(Counter(f"{w}x{h}" for w, h, _, _ in info).most_common(3)),
                "fps_min_max": [round(min(i[2] for i in info), 2), round(max(i[2] for i in info), 2)],
                "seconds_median_min_max": [round(statistics.median(i[3] for i in info), 1), round(min(i[3] for i in info), 1), round(max(i[3] for i in info), 1)],
                "with_audio_stream": f"{audio} of {len(audio_sample)}"}
        entry["stats"] = stats
        report[dataset] = entry
    duplicates = [names for names in hashes.values() if len(names) > 1]
    report["duplicate_content_groups"] = {"count": len(duplicates), "first": duplicates[:5]}
    (out_dir / "set_check.yaml").write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))


def disk_key_of(dataset: str, path: Path) -> str:
    from list_test_videos import disk_key
    return disk_key(dataset, path)


if __name__ == "__main__":
    main()
