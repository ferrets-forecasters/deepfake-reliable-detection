from __future__ import annotations

import subprocess

PATTERN = "train_queue|train_reference|export_predictions|analyze_training|watch_progress"
COMMAND = (
    "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match '" + PATTERN + "' } | "
    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $_.ProcessId }"
)


def main() -> None:
    """Stops the training queue, its trainings, the export and the watchers. Every training keeps its state file
    after each finished epoch, so `uv run python scripts/train_queue.py --max-active 3` resumes where it stopped."""
    stopped = subprocess.run(["powershell", "-NoProfile", "-Command", COMMAND], capture_output=True, text=True).stdout.split()
    print(f"stopped {len(stopped)} processes")


if __name__ == "__main__":
    main()
