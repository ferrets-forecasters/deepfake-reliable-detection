from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import DRIVE_URL, open_remote  # noqa: E402

from deepfake_detection.config import load_config  # noqa: E402
from deepfake_detection.data.drive_archives import load_drive_archives  # noqa: E402

EXTRA_ZENODO = {"openforensics": "5528418"}
HEADERS = {"Accept": "application/vnd.github+json", "User-Agent": "deepfake-release-check"}


def get_json(url: str, attempts: int = 4, accept: str = "application/vnd.github+json") -> dict:
    for attempt in range(attempts):
        response = requests.get(url, headers={**HEADERS, "Accept": accept}, timeout=60)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:
            return {"error": "404"}
        time.sleep(5 * (attempt + 1))
    return {"error": f"HTTP {response.status_code}"}


def zenodo(record: str) -> dict:
    d = get_json(f"https://zenodo.org/api/records/{record}", accept="application/json")
    if "error" in d:
        return d
    meta = d.get("metadata", {})
    return {"record": record, "title": meta.get("title"), "doi": d.get("doi"), "version": meta.get("version"), "published": meta.get("publication_date"),
            "license": (meta.get("license") or {}).get("id"), "files": [{"name": f["key"], "bytes": f["size"], "checksum": f.get("checksum")} for f in d.get("files", [])]}


def github(url: str) -> dict:
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)", url)
    if not match:
        return {"error": "not a GitHub URL"}
    owner, repo = match.group(1), match.group(2).removesuffix(".git")
    d = get_json(f"https://api.github.com/repos/{owner}/{repo}")
    if "error" in d:
        return d
    release = get_json(f"https://api.github.com/repos/{owner}/{repo}/releases/latest")
    return {"repo": f"{owner}/{repo}", "license": (d.get("license") or {}).get("spdx_id"), "default_branch": d.get("default_branch"), "pushed_at": d.get("pushed_at"),
            "latest_release": None if "error" in release else {"tag": release.get("tag_name"), "published": release.get("published_at")}}


def drive_archive(config, name: str, file_id: str, file_name: str) -> dict:
    """Byte size of the shared-Drive archive and, for a zip, a fingerprint of its content list (name, size and CRC-32 of every member, taken from the
    central directory by ranged requests): the same fingerprint of an official release would show the archive is that release."""
    if file_name.endswith(".rar"):
        index = config.paths.datasets_root / name / "labels" / "rar_index.csv"
        rows = sum(1 for _ in open(index, encoding="utf-8")) - 1 if index.exists() else None
        head = requests.get(DRIVE_URL.format(id=file_id), headers={"Range": "bytes=0-0"}, timeout=60)
        return {"file": file_name, "bytes": int(head.headers["Content-Range"].split("/")[1]), "members_indexed_locally": rows, "fingerprint": None}
    archive = open_remote(DRIVE_URL.format(id=file_id))
    infos = [i for i in archive.infolist() if not i.is_dir()]
    listing = "\n".join(f"{i.filename}\t{i.file_size}\t{i.CRC:08x}" for i in sorted(infos, key=lambda i: i.filename))
    top = Counter("/".join(Path(i.filename).parts[:2]) for i in infos)
    return {"file": file_name, "bytes": archive.fp.raw.size, "members": len(infos), "uncompressed_bytes": sum(i.file_size for i in infos),
            "fingerprint_sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(), "members_by_top_folders": dict(top.most_common(12))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Release facts of the datasets: Zenodo version, license and md5 of the files, GitHub license and last push, and for each archive in the shared Drive its byte size and a fingerprint of its member list. One cache file per source, so a rerun only asks what is missing.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.results_dir("dataset_releases")
    cache = out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "progress.log", "a", encoding="utf-8")

    with open(config.paths.datasets_root / "manifest.csv", encoding="utf-8", newline="") as f:
        registry = {r["name"]: r for r in csv.DictReader(f)}
    drive_archives = load_drive_archives(required=False)
    jobs: list[tuple[str, str, object]] = []
    for name, row in registry.items():
        jobs.append((name, "github", lambda url=row["source_url"]: github(url)))
        zen = re.search(r"zenodo\.org/records/(\d+)", row["download_url"])
        if zen:
            jobs.append((name, "zenodo", lambda rec=zen.group(1): zenodo(rec)))
        if len(drive_archives.get(name, [])) == 1:
            file_id, file_name, _ = drive_archives[name][0]
            jobs.append((name, "shared_drive_archive", lambda n=name, i=file_id, fn=file_name: drive_archive(config, n, i, fn)))
    for name, record in EXTRA_ZENODO.items():
        jobs.append((name, "zenodo", lambda rec=record: zenodo(rec)))

    report: dict = {}
    for name, kind, run in tqdm(jobs, desc="sources", unit="source"):
        path = cache / f"{name}_{kind}.json"
        if path.exists() and not args.refresh:
            result = json.loads(path.read_text(encoding="utf-8"))
        else:
            start = time.time()
            try:
                result = run()
            except Exception as error:
                result = {"error": f"{type(error).__name__}: {error}"}
            if "error" not in result:
                path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            log.write(f"{time.strftime('%T')} {name} {kind} {time.time() - start:.1f} s {'error ' + result['error'] if 'error' in result else 'ok'}\n")
            log.flush()
        report.setdefault(name, {})[kind] = result

    (out_dir / "releases.yaml").write_text(yaml.safe_dump(report, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
