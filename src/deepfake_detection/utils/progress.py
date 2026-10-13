from __future__ import annotations

import csv
from pathlib import Path

from tqdm import tqdm

BAR_FORMAT = "{desc} {n_fmt}/{total_fmt} {percentage:3.0f}% elapsed {elapsed} eta {remaining} {rate_fmt}"


class _LineFile:
    """File-like object for tqdm: each refresh becomes one flushed line of a log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, text: str) -> int:
        text = text.replace("\r", "").strip()
        if text:
            self._handle.write(text + "\n")
            self._handle.flush()
        return len(text)

    def flush(self) -> None:
        self._handle.flush()


def progress(total: int, done: int, log_path: Path, desc: str, interval_seconds: float = 20.0) -> tqdm:
    """A bar that resumes at `done` and writes one line per refresh, with elapsed time and ETA, to log_path."""
    return tqdm(total=total, initial=done, desc=desc, file=_LineFile(log_path), bar_format=BAR_FORMAT, mininterval=interval_seconds, miniters=1)


class ResumableCsv:
    """Append-only CSV of finished units. `done` holds the keys already written, so a re-run skips them."""

    def __init__(self, path: Path, columns: list[str], key: str):
        self.path, self.columns, self.key = path, columns, key
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        else:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=columns).writeheader()
        self.done = {row[key] for row in self.rows}

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow(row)
        self.rows.append({k: str(v) for k, v in row.items()})
        self.done.add(str(row[self.key]))
