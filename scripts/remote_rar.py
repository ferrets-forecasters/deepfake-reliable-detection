from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
import time
import zlib
from pathlib import Path

import requests
from tqdm import tqdm

SIGNATURE = b"Rar!\x1a\x07\x01\x00"
HEADER_WINDOW = 4096
PROBE_WINDOW = 2_000_000
STOP_SPAN = 4_000_000
FILE_HEADER, END_HEADER = 2, 5
NAME_PATTERN = re.compile(rb"[A-Za-z0-9_]+/(?:real|fake)/[^\x00]*?\.mp4")


def vint(buffer: bytes, position: int) -> tuple[int, int]:
    """RAR5 variable-length integer: 7 bits per byte, low bits first, high bit set while more bytes follow."""
    value = shift = 0
    while True:
        byte = buffer[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


def read_range(session: requests.Session, url: str, start: int, end: int, attempts: int = 6) -> bytes:
    for attempt in range(attempts):
        try:
            response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if response.status_code == 206:
                return response.content
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"range {start}-{end} failed")


def parse_block(buffer: bytes, start: int, offset: int) -> dict | None:
    """The block that starts at buffer[start] and sits at archive position offset. Returns None when the
    header checksum fails or the block is cut off; a dict with next_offset, and name for file blocks."""
    try:
        size, position = vint(buffer, start + 4)
        body_end = position + size
        if body_end > len(buffer) or zlib.crc32(buffer[start + 4:body_end]) != int.from_bytes(buffer[start:start + 4], "little"):
            return None
        block_type, position = vint(buffer, position)
        flags, position = vint(buffer, position)
        if flags & 1:
            _, position = vint(buffer, position)
        data_size = 0
        if flags & 2:
            data_size, position = vint(buffer, position)
        block = {"offset": offset, "next_offset": offset + (body_end - start) + data_size, "data_size": data_size, "type": block_type}
        if block_type != FILE_HEADER:
            return block
        file_flags, position = vint(buffer, position)
        _, position = vint(buffer, position)
        _, position = vint(buffer, position)
        position += 4 * bool(file_flags & 2) + 4 * bool(file_flags & 4)
        _, position = vint(buffer, position)
        _, position = vint(buffer, position)
        name_length, position = vint(buffer, position)
        block["name"] = buffer[position:position + name_length].decode("utf-8", "replace")
        return block
    except IndexError:
        return None


def header_near(session: requests.Session, url: str, offset: int) -> dict | None:
    """The first complete file header found in a window that starts at offset, located by its path string.
    The window grows while it holds no header, because one video can be larger than the first window."""
    for width in (PROBE_WINDOW, 4 * PROBE_WINDOW, 16 * PROBE_WINDOW):
        window = read_range(session, url, offset, offset + width - 1)
        for match in NAME_PATTERN.finditer(window):
            for start in range(max(0, match.start() - 64), match.start()):
                block = parse_block(window, start, offset + start)
                if block and block.get("name", "").encode() == match.group(0):
                    return block
    return None


def locate(session: requests.Session, url: str, total: int, target: str, first_offset: int, fetch_header) -> dict | None:
    """Binary search over archive positions for a member, relying on the archive listing paths in sorted order,
    then a short walk from the nearest header at or before it."""
    anchor, low, high = first_offset, first_offset, total
    while high - low > STOP_SPAN:
        middle = (low + high) // 2
        block = fetch_header(middle)
        if block is None:
            low = middle + 16 * PROBE_WINDOW
        elif block["name"] <= target:
            if block["offset"] <= low:
                break
            anchor, low = block["offset"], block["offset"]
        else:
            if block["offset"] >= high:
                break
            high = block["offset"]
    offset = anchor
    while offset < total:
        block = parse_block(read_range(session, url, offset, offset + HEADER_WINDOW - 1), 0, offset)
        if block is None or block["type"] == END_HEADER:
            return None
        if block.get("name") == target:
            return block
        if block.get("name", "") > target:
            return None
        offset = block["next_offset"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download chosen members of a remote RAR5 archive with ranged requests: each member is found by a binary search on the sorted listing and extracted from a small archive built from its own bytes.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--member", nargs="+", required=True, help="exact member paths inside the archive")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unrar", default=r"C:\Program Files\WinRAR\UnRAR.exe", help="RAR extractor; bsdtar leaves the end of some compressed members as zeros")
    args = parser.parse_args()

    session = requests.Session()
    total = int(session.get(args.url, headers={"Range": "bytes=0-0"}, timeout=60).headers["Content-Range"].split("/")[1])
    head = read_range(session, args.url, 0, HEADER_WINDOW - 1)
    if head[:len(SIGNATURE)] != SIGNATURE:
        raise SystemExit("not an RAR5 archive")
    main_block = parse_block(head, len(SIGNATURE), len(SIGNATURE))
    prefix = head[:main_block["next_offset"]]

    for member in tqdm(args.member, desc="members"):
        target = args.out / member
        if target.exists():
            continue
        block = locate(session, args.url, total, member, main_block["next_offset"], lambda at: header_near(session, args.url, at))
        if block is None:
            print(f"not found: {member}")
            continue
        data = read_range(session, args.url, block["offset"], block["next_offset"] - 1)
        with tempfile.TemporaryDirectory(dir=args.out.parent if args.out.parent.exists() else None) as scratch:
            archive = Path(scratch) / "one.rar"
            archive.write_bytes(prefix + data)
            args.out.mkdir(parents=True, exist_ok=True)
            subprocess.run([args.unrar, "x", "-y", "-inul", str(archive), f"{args.out}{os.sep}"], check=False)
            if not target.exists():
                print(f"extraction failed: {member}")


if __name__ == "__main__":
    main()
