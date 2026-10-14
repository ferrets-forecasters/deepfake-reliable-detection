from __future__ import annotations

import faulthandler
from contextlib import contextmanager
from typing import TextIO


@contextmanager
def hang_watchdog(seconds: float, log: TextIO, label: str = ""):
    """Writes the traceback of every thread to `log` when the block runs longer than `seconds`, so a stall names the line it is stuck on.
    The block keeps running; stopping a stalled step is the caller's timeout."""
    log.write(f"watchdog armed {seconds:.0f} s {label}\n")
    log.flush()
    faulthandler.dump_traceback_later(seconds, repeat=False, file=log)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
