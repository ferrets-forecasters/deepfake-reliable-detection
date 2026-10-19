from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_tool(name: str, explicit: str | Path | None = None) -> Path:
    """Path of an external program: the explicit path if given, else the environment variable <NAME>_PATH, else the first match on PATH."""
    found = explicit or os.environ.get(f"{name.upper()}_PATH") or shutil.which(name)
    if not found:
        raise FileNotFoundError(f"{name} not found on PATH; install it, pass its path, or set {name.upper()}_PATH")
    return Path(found)
