from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
BLOCK = 4 << 20


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, with a one-block read-ahead cache, so zipfile can
    read the central directory and single members of a remote archive without downloading all of it."""

    def __init__(self, url: str, attempts: int = 6):
        self.url, self.attempts, self.session = url, attempts, requests.Session()
        head = self.session.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
        self.size = int(head.headers["Content-Range"].split("/")[1])
        self.position, self.block_start, self.block = 0, 0, b""

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(self.attempts):
            try:
                response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
                if response.status_code == 206:
                    return response.content
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"range {start}-{end} failed")

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = {io.SEEK_SET: offset, io.SEEK_CUR: self.position + offset, io.SEEK_END: self.size + offset}[whence]
        return self.position

    def readinto(self, buffer) -> int:
        want = min(len(buffer), self.size - self.position)
        if want <= 0:
            return 0
        offset = self.position - self.block_start
        if not 0 <= offset or offset + want > len(self.block):
            self.block_start = self.position
            self.block = self._fetch(self.position, min(self.position + max(want, BLOCK) - 1, self.size - 1))
            offset = 0
        buffer[:want] = self.block[offset:offset + want]
        self.position += want
        return want


def open_remote(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 16))


def main() -> None:
    parser = argparse.ArgumentParser(description="List or extract members of a remote zip archive (a Drive file id or a URL) with ranged requests.")
    parser.add_argument("--url", help="archive URL")
    parser.add_argument("--drive-id", help="Drive file id instead of a URL")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--out", type=Path, required=True, help="text file, one 'size<TAB>name' line per member")
    extract = sub.add_parser("extract")
    extract.add_argument("--member", nargs="+", required=True)
    extract.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    url = args.url or DRIVE_URL.format(id=args.drive_id)
    archive = open_remote(url)
    if args.command == "list":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(f"{i.file_size}\t{i.filename}" for i in archive.infolist() if not i.is_dir()), encoding="utf-8")
        print(f"{len(archive.infolist())} members listed in {args.out}")
        return
    for name in tqdm(args.member, desc="members"):
        target = args.out / name
        if target.exists():
            continue
        try:
            info = archive.getinfo(name)
        except KeyError:
            print(f"not found: {name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as src, open(target, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)


if __name__ == "__main__":
    main()
