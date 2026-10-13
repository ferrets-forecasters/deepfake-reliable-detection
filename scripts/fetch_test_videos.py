from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import gdown
from tqdm import tqdm

from deepfake_detection.config import load_config

# Own split: the combined test split of the retrained gated model (seed 42, 20% of the subjects of every dataset).
OWN_PREDICTIONS = "moe_combined_retrain_on_combined_test_seed42"
# Drive file ids in the shared folder 1UZ61KBsorHL0WdaLaCklDEG2-B17oenl, smallest first.
ARCHIVES = {
    "hidf": [("1mB1NHKuJiBjgyW9OqnxcnlqJ85qCrIaa", "Real-vid.zip"), ("1UTfL7rvWwR9-__oHolDnxbWKsBXFBut1", "Fake-vid.zip")],
    "fakeavceleb": [("1LlDGdHKx3LVs2gJaY_7qAsLPwW3bPyv8", "FAKEAVCELEB.zip")],
    "celebdfv2": [("1ycE03-UpDrrTKfem5FSewMNf5mtHL5Cw", "Celeb-DF-v2.zip")],
    "ffpp": [("18M8bOxbPT_d6Zii7Ah2HAV7XgG0_piY_", "FaceForensics++ Dataset (C23).zip")],
    "polyglotfake": [("1xH5DTDHghOsP3R8pQLUuSz_MfKY60yc5", "PolyGlotFake.rar")],
}
ATTEMPTS = 6
FFPP_METHODS = ("deepfakes", "face2face", "faceshifter", "faceswap", "neuraltextures", "deepfakedetection")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}


def own_test_ids(config) -> dict[str, dict[str, dict]]:
    """dataset -> sample id -> label, type and subject of every video in the own test split."""
    path = config.paths.predictions_dir(OWN_PREDICTIONS) / "predictions.csv"
    own: dict[str, dict[str, dict]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sample = re.sub(r"_seg\d+$", "", Path(row["video"]).stem)
            own.setdefault(row["dataset"], {})[sample] = {"label": row["label"], "manip_type": row["manip_type"], "subject": row["subject"]}
    return own


def key_of(dataset: str, member: str) -> str:
    """The sample id our embeddings use for an archive member."""
    path = Path(member)
    if dataset == "ffpp":
        parts = [p.lower() for p in path.parts]
        method = next((p for p in parts if p in FFPP_METHODS), None)
        return f"{method}_{path.stem}" if method else path.stem
    return path.stem


@lru_cache(maxsize=1)
def ffpp_test_ids(path: Path) -> frozenset:
    return frozenset(x for pair in json.loads(path.read_text(encoding="utf-8")) for x in pair)


def ffpp_official(config, member: str) -> bool:
    ids = ffpp_test_ids(config.paths.datasets_root / "ffpp" / "labels" / "official_split" / "test.json")
    parts = [p.lower() for p in Path(member).parts]
    if "deepfakedetection" in parts:
        return False
    found = re.findall(r"\d{3}", Path(member).stem)
    return bool(found) and all(i in ids for i in found[:2])


def ffpp_meta(member: str) -> dict:
    """Label and manipulation type of an official-test member that is not in the own split."""
    method = next((p for p in (x.lower() for x in Path(member).parts) if p in FFPP_METHODS), None)
    return {"label": "1" if method else "0", "manip_type": method or "real", "subject": ""}


FIELDS = ["member", "sample_id", "label", "manip_type", "subject", "own_test_seed42", "official_test", "stem_candidates"]


def annotate_candidates(rows: list[dict]) -> list[dict]:
    """Embedding files are named by the video's file stem, and a stem can occur in several folders of the
    release (FakeAVCeleb repeats clip numbers across identities). stem_candidates counts the archive members
    that carry the sample's stem: 1 means the video is certain, more means one of them was embedded."""
    counts: dict[str, int] = {}
    for row in rows:
        if row["sample_id"]:
            counts[row["sample_id"]] = counts.get(row["sample_id"], 0) + 1
    for row in rows:
        row["stem_candidates"] = counts[row["sample_id"]] if row["sample_id"] else ""
    return rows


def download(dataset: str, file_id: str, archive: Path) -> bool:
    """Downloads one Drive file, resuming a partial copy. False when Drive refuses it (its daily download
    quota), so the caller moves on to the next dataset; other errors are retried."""
    print(f"{dataset}: downloading {archive.name}", flush=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        try:
            gdown.download(id=file_id, output=str(archive), quiet=False, resume=True)
            return True
        except gdown.exceptions.FileURLRetrievalError as error:
            print(f"{dataset}: {archive.name} refused by Drive: {str(error).splitlines()[-1][:120]}", flush=True)
            return False
        except Exception as error:
            print(f"{dataset}: {archive.name} attempt {attempt} of {ATTEMPTS} failed ({type(error).__name__}), resuming", flush=True)
            time.sleep(10)
    return False


def list_members(archive: Path) -> list[str]:
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            return [n for n in z.namelist() if not n.endswith("/")]
    listing = subprocess.run(["tar", "-tf", str(archive)], capture_output=True, text=True, check=True).stdout
    return [n for n in listing.splitlines() if n and not n.endswith("/")]


def extract(archive: Path, members: list[str], target: Path) -> None:
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in tqdm(members, desc=f"extract {archive.name}", unit="file", mininterval=20):
                out = target / name
                if out.exists() and out.stat().st_size == z.getinfo(name).file_size:
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
        return
    listing = target / ".members.txt"
    target.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(members), encoding="utf-8")
    subprocess.run(["tar", "-xf", str(archive), "-C", str(target), "-T", str(listing)], check=True)
    listing.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Per dataset: download its archives from the shared Drive, extract the videos of the own test split and of the official test split into datasets/<name>/raw/test_split/, write labels/test_split.csv, then delete the archives. Run with `uv run --with gdown python scripts/fetch_test_videos.py`. A dataset with a finished test_split.csv is skipped.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--datasets", nargs="*", default=list(ARCHIVES))
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--annotate-only", action="store_true", help="add the stem_candidates column to the finished test_split.csv files and stop")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.annotate_only:
        for dataset in args.datasets:
            table = config.paths.datasets_root / dataset / "labels" / "test_split.csv"
            if not table.exists():
                continue
            with open(table, encoding="utf-8") as f:
                rows = annotate_candidates(list(csv.DictReader(f)))
            with open(table, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"{dataset}: annotated {len(rows)} rows", flush=True)
        return
    own = own_test_ids(config)
    for dataset in args.datasets:
        raw = config.raw_dir(dataset)
        table = config.paths.datasets_root / dataset / "labels" / "test_split.csv"
        if table.exists():
            print(f"{dataset}: test_split.csv present, skipped", flush=True)
            continue
        wanted = own.get(dataset, {})
        rows: dict[str, dict] = {}
        matched = set()
        for file_id, name in ARCHIVES[dataset]:
            archive = raw / name
            if not archive.exists() and not download(dataset, file_id, archive):
                break
            members = [m for m in list_members(archive) if Path(m).suffix.lower() in VIDEO_SUFFIXES]
            print(f"{dataset}: {name} holds {len(members)} videos", flush=True)
            chosen = []
            for member in members:
                key = key_of(dataset, member)
                in_own = key in wanted
                in_official = dataset == "ffpp" and ffpp_official(config, member)
                if in_own:
                    matched.add(key)
                if in_own or in_official:
                    meta = wanted.get(key) or ffpp_meta(member)
                    rows[member] = {"member": member, "sample_id": key if in_own else "", "label": meta.get("label", ""), "manip_type": meta.get("manip_type", ""),
                                    "subject": meta.get("subject", ""), "own_test_seed42": int(in_own), "official_test": int(in_official)}
                    chosen.append(member)
            extract(archive, chosen, raw / "test_split")
            if dataset == "celebdfv2":
                lists = [n for n in zipfile.ZipFile(archive).namelist() if n.endswith("List_of_testing_videos.txt")]
                if lists:
                    out = config.paths.datasets_root / dataset / "labels" / "official_split"
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "List_of_testing_videos.txt").write_bytes(zipfile.ZipFile(archive).read(lists[0]))
        else:
            missing = sorted(set(wanted) - matched)
            print(f"{dataset}: own test videos {len(wanted)}, matched {len(matched)}, not found in the archives {len(missing)}", flush=True)
            table.parent.mkdir(parents=True, exist_ok=True)
            with open(table, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(annotate_candidates(list(rows.values())))
            if missing:
                (table.parent / "test_split_not_found.txt").write_text("\n".join(missing), encoding="utf-8")
            if not args.keep_archives:
                for _, name in ARCHIVES[dataset]:
                    (raw / name).unlink(missing_ok=True)
            continue
        print(f"{dataset}: archive not available now, skipped", flush=True)


if __name__ == "__main__":
    main()
