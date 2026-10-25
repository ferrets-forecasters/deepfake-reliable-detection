from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.drive_archives import load_drive_archives

# The combined test split of the retrained gated model: seed 42, 20% of the subjects of every dataset.
OWN_PREDICTIONS = "moe_combined_retrain_on_combined_test_seed42"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
FFPP_FOLDERS = {"deepfakes": "Deepfakes", "face2face": "Face2Face", "faceshifter": "FaceShifter", "faceswap": "FaceSwap",
                "neuraltextures": "NeuralTextures", "deepfakedetection": "DeepFakeDetection"}
FETCH_COLUMNS = ["dataset", "archive", "archive_member", "label", "splits"]
COLUMNS = ["dataset", "sample_id", "label", "manip_type", "subject", "on_disk", "local_path", "archive", "archive_member", "member_source"]


def own_test_rows(config) -> list[dict]:
    path = config.paths.predictions_dir(OWN_PREDICTIONS) / "predictions.csv"
    with open(path, encoding="utf-8") as f:
        return [{**r, "sample_id": re.sub(r"_seg\d+$", "", Path(r["video"]).stem)} for r in csv.DictReader(f)]


def listed_members(config, dataset: str) -> dict[str, list[str]]:
    """Archive members that the repository has already listed: sample id -> members (HiDF and FakeAVCeleb test_split.csv, PolyGlotFake rar_index.csv)."""
    labels = config.labels_dir(dataset)
    members: dict[str, list[str]] = {}
    if (labels / "test_split.csv").exists():
        with open(labels / "test_split.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["own_test_seed42"] == "1":
                    members.setdefault(row["sample_id"], []).append(row["member"])
    if (labels / "rar_index.csv").exists():
        with open(labels / "rar_index.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                members.setdefault(Path(row["name"]).stem, []).append(row["name"])
    return members


def convention_member(dataset: str, sample: str, manip_type: str) -> str:
    """Member path from the dataset's naming convention, for a video that no listing names."""
    if dataset == "polyglotfake":
        if sample.startswith("fake_"):
            name = sample.removeprefix("fake_")
            return f"PolyGlotFake/fake/to_{re.search(r'_to_([a-z]+)_', name).group(1)}/{name}.mp4"
        return f"PolyGlotFake/real/{sample}.mp4"
    if dataset == "celebdfv2":
        ids = len(re.findall(r"id\d+", sample))
        folder = "Celeb-synthesis" if ids == 2 else "Celeb-real" if ids == 1 else "YouTube-real"
        return f"{folder}/{sample}.mp4"
    if dataset == "ffpp":
        method = next((m for m in FFPP_FOLDERS if sample.startswith(m + "_")), None)
        if method:
            return f"manipulated_sequences/{FFPP_FOLDERS[method]}/c23/videos/{sample.removeprefix(method + '_')}.mp4"
        return f"original_sequences/youtube/c23/videos/{sample}.mp4"
    return ""


def disk_key(dataset: str, path: Path) -> str:
    """The sample id our embeddings use for a raw file: FF++ names repeat in every method folder, so the method leads."""
    if dataset == "ffpp":
        method = next((part.lower() for part in path.parts if part.lower() in FFPP_FOLDERS), None)
        return f"{method}_{path.stem}" if method else path.stem
    if dataset == "polyglotfake" and "fake" in path.parts:  # the fake files are named <real name>_to_<language>_<voice>, our sample ids lead with fake_
        return f"fake_{path.stem}"
    return path.stem


def disk_index(dataset: str, raw: Path) -> dict[str, Path]:
    """Sample id -> path of every video under a dataset's raw folder."""
    return {disk_key(dataset, p): p for p in raw.rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES and not p.name.endswith(".part")}


def official_rows(config, dataset: str) -> list[dict]:
    """The official test split of a dataset where one exists: FF++ (test.json pairs, five manipulation types) and Celeb-DF v2 (List_of_testing_videos.txt)."""
    rows = []
    if dataset == "ffpp":
        pairs = json.loads((config.labels_dir(dataset) / "official_split" / "test.json").read_text(encoding="utf-8"))
        for a, b in pairs:
            for sample in (a, b):
                rows.append({"sample_id": sample, "label": "0", "manip_type": "real", "member": f"original_sequences/youtube/c23/videos/{sample}.mp4"})
            for key, folder in FFPP_FOLDERS.items():
                if key == "deepfakedetection":
                    continue
                for src, dst in ((a, b), (b, a)):
                    rows.append({"sample_id": f"{key}_{src}_{dst}", "label": "1", "manip_type": key, "member": f"manipulated_sequences/{folder}/c23/videos/{src}_{dst}.mp4"})
    elif dataset == "celebdfv2":
        listing = config.raw_dir(dataset) / "test_official" / "List_of_testing_videos.txt"
        if listing.exists():
            for line in listing.read_text(encoding="utf-8").splitlines():
                flag, member = line.split(maxsplit=1)
                rows.append({"sample_id": Path(member).stem, "label": "0" if flag == "1" else "1", "manip_type": "real" if flag == "1" else "fake", "member": member})
    return rows


def mean_mb(paths) -> float:
    sizes = [p.stat().st_size for p in paths if p.is_file()]
    return statistics.mean(sizes) / 1e6


def write_handoff(config, out_dir: Path, fetch: list[dict]) -> None:
    """FETCH_TEST_VIDEOS.md: what an agent on another laptop needs to cut the missing test videos out of the shared Drive archives and zip them, with a size estimate."""
    releases = yaml.safe_load((config.paths.results_dir("dataset_releases").parent / "dataset_releases" / "releases.yaml").read_text(encoding="utf-8"))
    datasets = config.paths.datasets_root
    local = {name: [p for p in (datasets / name / "raw").rglob("*") if p.suffix.lower() in VIDEO_SUFFIXES and not p.name.endswith(".part")] for name in ("celebdfv2", "ffpp", "polyglotfake")}
    with open(config.labels_dir("polyglotfake") / "rar_index.csv", encoding="utf-8") as f:
        rar_mb = statistics.mean(int(r["unpacked_size"]) for r in csv.DictReader(f)) / 1e6
    lines, low, high = [], 0.0, 0.0
    for name in ("polyglotfake", "ffpp", "celebdfv2"):
        n = sum(r["dataset"] == name for r in fetch)
        archive = releases[name]["shared_drive_archive"]
        by_archive = archive["bytes"] / archive["members"] / 1e6 if archive.get("members") else rar_mb
        a, b = sorted((by_archive, mean_mb(local[name])))
        low, high = low + n * a / 1e3, high + n * b / 1e3
        lines.append(f"| {name} | `{archive['file']}` | {archive['bytes'] / 1e9:.1f} GB | {n} | {a:.1f} to {b:.1f} MB | {n * a / 1e3:.1f} to {n * b / 1e3:.1f} GB |")
    table = chr(10).join(lines)
    text = f"""# Zip the missing test videos

For an agent on a laptop with access to the team's shared Google Drive. Rifqi needs {len(fetch)} raw test videos (listed in `fetch_list.csv`, next to this file) that sit inside three archives on the shared Drive. Cut exactly those files out of the archives and zip them. Do not train, convert, rename or re-encode anything, and do not upload the result anywhere unless Rifqi says where.

## Input
`fetch_list.csv`, one row per video: `dataset`, `archive` (file name on the shared Drive), `archive_member` (path of the video inside the archive), `label` (0 real, 1 fake), `splits` (which test split needs it, for information only). The member paths follow each dataset's usual folder layout and were not read from the archives, so a path may differ a little: FF++ `original_sequences/youtube/c23/videos/<id>.mp4` and `manipulated_sequences/<Method>/c23/videos/<id>_<id>.mp4`; Celeb-DF v2 `Celeb-real/`, `YouTube-real/` and `Celeb-synthesis/`; PolyGlotFake `PolyGlotFake/real/<name>.mp4` and `PolyGlotFake/fake/to_<language>/<name>.mp4`.

## Steps
1. Download each archive from the shared Drive into an empty folder, one at a time. If Drive refuses a download (daily quota), stop and tell Rifqi which archive; do not use another source.
2. Take only the members of that archive listed in the CSV. `.zip` archives: Python `zipfile`. `PolyGlotFake.rar`: 7-Zip, or Windows `tar -xf <archive> -T <list.txt>` with the member paths one per line.
3. A path that is not in the archive: look for a member with the same file name (without folder) and use it; if there is none or there are several, leave it out and list it in the report.
4. Keep the member path inside the zip. Output, in a new empty folder, one zip per dataset (store, no compression, the videos are already compressed): `polyglotfake.zip`, `ffpp.zip`, `celebdfv2.zip`. Each zip holds `<archive_member>` as its entry path, for example `ffpp.zip` holds `manipulated_sequences/Deepfakes/c23/videos/000_003.mp4`.
5. Delete the downloaded archive after its zip is checked (the archives are about 55 GB together).
6. Write `report.txt` in the output folder: per dataset, the rows in the CSV, the files zipped, the paths left out, and the size of the zip.

## Size
Mp4 does not compress, so each zip is about the sum of its videos. Estimate from the average video size (archive size divided by its member count, and the average of the videos Rifqi already has):

| Dataset | Archive | Archive size | Videos to zip | Average per video | Zip size |
|---|---|---|---|---|---|
{table}
| Total | | | {len(fetch)} | | {low:.1f} to {high:.1f} GB |

Free disk needed: the largest archive (26.7 GB) plus all three zips ({high:.0f} GB at most), so about 50 GB if the archive is deleted after each zip, 75 GB if all three archives stay. The download of 55 GB is the slow part.
"""
    (out_dir / "FETCH_TEST_VIDEOS.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="List every video of the own seed-42 test split per dataset with whether the raw file is on disk, the Drive archive that holds it and its member path, so the missing videos can be fetched. Writes results/analyses/<name>/<dataset>.csv, missing_<dataset>.txt and summary.yaml.")
    parser.add_argument("--name", default="test_video_inventory")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    archives = load_drive_archives(required=False)
    out_dir = config.paths.analysis_dir(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_rows = own_test_rows(config)
    summary: dict[str, dict] = {}
    fetch: list[dict] = []
    for dataset in tqdm(config.dataset_names(), desc="datasets"):
        wanted = [r for r in test_rows if r["dataset"] == dataset]
        on_disk = disk_index(dataset, config.raw_dir(dataset))
        listed = listed_members(config, dataset)
        names = [name for _, name, _ in archives.get(dataset, [])]
        rows = []
        for r in wanted:
            sample = r["sample_id"]
            members = listed.get(sample) or [convention_member(dataset, sample, r["manip_type"])]
            source = "listed" if sample in listed else "convention" if members[0] else "unknown"
            path = on_disk.get(sample)
            rows.append({"dataset": dataset, "sample_id": sample, "label": r["label"], "manip_type": r["manip_type"], "subject": r["subject"],
                         "on_disk": int(path is not None), "local_path": str(path.relative_to(config.paths.datasets_root)) if path else "",
                         "archive": " | ".join(names), "archive_member": " | ".join(m for m in members if m), "member_source": source})
        with open(out_dir / f"{dataset}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        official = [{"dataset": dataset, "sample_id": o["sample_id"], "label": o["label"], "manip_type": o["manip_type"], "subject": "", "on_disk": int(o["sample_id"] in on_disk),
                     "local_path": str(on_disk[o["sample_id"]].relative_to(config.paths.datasets_root)) if o["sample_id"] in on_disk else "",
                     "archive": " | ".join(names), "archive_member": o["member"], "member_source": "official list"} for o in official_rows(config, dataset)]
        if official:
            with open(out_dir / f"{dataset}_official.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(official)
        missing = [r for r in rows if not r["on_disk"]]
        for split, group in (("own seed-42 test", rows), ("official test", official)):
            fetch.extend({"dataset": dataset, "split": split, "archive": r["archive"], "archive_member": r["archive_member"], "label": r["label"], "member_source": r["member_source"]}
                         for r in group if not r["on_disk"])
        (out_dir / f"missing_{dataset}.txt").write_text("\n".join(r["archive_member"] for r in missing), encoding="utf-8")
        summary[dataset] = {"test_videos": len(rows), "on_disk": len(rows) - len(missing), "missing": len(missing),
                            "missing_real": sum(r["label"] == "0" for r in missing), "missing_fake": sum(r["label"] == "1" for r in missing),
                            "member_source_of_missing": dict(Counter(r["member_source"] for r in missing)), "archive": " | ".join(names), "archive_gb": round(sum(b or 0 for _, _, b in archives.get(dataset, [])) / 1e9, 1), "official_test": len(official), "official_missing": sum(not o["on_disk"] for o in official)}
    members: dict[tuple, dict] = {}
    for r in fetch:
        row = members.setdefault((r["dataset"], r["archive_member"]), {"dataset": r["dataset"], "archive": r["archive"], "archive_member": r["archive_member"],
                                                                     "label": r["label"], "splits": set()})
        row["splits"].add("official" if r["split"] == "official test" else "own")
    fetch = [{**r, "splits": "+".join(sorted(r["splits"]))} for r in members.values()]
    with open(out_dir / "fetch_list.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FETCH_COLUMNS)
        writer.writeheader()
        writer.writerows(fetch)
    write_handoff(config, out_dir, fetch)
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    for dataset, s in summary.items():
        print(f"{dataset}: {s['test_videos']} test videos, {s['on_disk']} on disk, {s['missing']} missing")


if __name__ == "__main__":
    main()
