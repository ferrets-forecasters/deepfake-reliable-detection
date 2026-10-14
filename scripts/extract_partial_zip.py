from __future__ import annotations

import argparse
import re
import struct
import zlib
from pathlib import Path

from tqdm import tqdm

LOCAL_HEADER = b"PK\x03\x04"


def entries(handle, size: int):
    """Yields (name, compression, compressed size, data offset) for every complete entry read
    from the local headers, in file order. Stops at the first truncated or streamed entry."""
    offset = 0
    while offset + 30 <= size:
        handle.seek(offset)
        header = handle.read(30)
        if header[:4] != LOCAL_HEADER:
            return
        _, flags, compression, _, _, _, compressed, _, name_length, extra_length = struct.unpack("<HHHHHIIIHH", header[4:30])
        name = handle.read(name_length).decode("utf-8", "replace")
        data_offset = offset + 30 + name_length + extra_length
        if (flags & 8 and compressed == 0) or data_offset + compressed > size:
            return
        yield name, compression, compressed, data_offset
        offset = data_offset + compressed


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract members from a zip archive that was only partly downloaded, by reading its local headers in order.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pattern", default=".*", help="regular expression matched against the member name")
    parser.add_argument("--list", action="store_true", help="print the matching member names and stop")
    args = parser.parse_args()

    wanted = re.compile(args.pattern)
    size = args.archive.stat().st_size
    with open(args.archive, "rb") as handle:
        members = [e for e in entries(handle, size) if not e[0].endswith("/") and wanted.search(e[0])]
        if args.list:
            print(*(m[0] for m in members), sep="\n")
            return
        for name, compression, compressed, data_offset in tqdm(members, desc="members"):
            target = args.out / name
            if target.exists():
                continue
            handle.seek(data_offset)
            raw = handle.read(compressed)
            data = zlib.decompress(raw, -15) if compression == 8 else raw
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print(f"{len(members)} members in {args.out}")


if __name__ == "__main__":
    main()
