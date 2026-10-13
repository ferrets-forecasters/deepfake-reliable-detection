from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT

FRESH_SECONDS = 600
STALL_SECONDS = 1200
MAX_ATTEMPTS = 3


def age_seconds(path: Path) -> float:
    return time.time() - path.stat().st_mtime if path.exists() else float("inf")


def is_finished(folder: Path) -> bool:
    return (folder / "results.yaml").exists()


def process_alive(pid: int, name: str) -> bool:
    """The process of this run is alive: windows reuses process ids, so the command line must name the run."""
    command = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"
    output = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True).stdout
    return f"--name {name} " in output + " "


def is_active(folder: Path) -> bool:
    """A run some process is training: no results yet, and either the process named in its claim file is alive
    or it wrote its progress file in the last 10 minutes."""
    if is_finished(folder):
        return False
    claim = folder / "claim"
    if claim.exists() and claim.read_text(encoding="utf-8").strip().isdigit() and process_alive(int(claim.read_text(encoding="utf-8").strip()), folder.name):
        return True
    return age_seconds(folder / "progress.log") < FRESH_SECONDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trainings in configs/training_queue.yaml, up to --max-active at the same time across all queue processes. Finished runs are skipped, runs another process is training are left alone, and an interrupted run resumes at its next epoch.")
    parser.add_argument("--queue", default=str(REPO_ROOT / "configs" / "training_queue.yaml"))
    parser.add_argument("--only", nargs="*", default=None, help="run only these names")
    parser.add_argument("--max-active", type=int, default=2, help="trainings active at once, counting runs started by other queue processes")
    parser.add_argument("--project-config", default=None)
    args = parser.parse_args()

    config = load_config(args.project_config)
    queue = yaml.safe_load(Path(args.queue).read_text(encoding="utf-8"))
    runs = [r for r in queue["runs"] if not args.only or r["name"] in args.only]
    script = Path(__file__).with_name("train_reference.py")
    folders = {r["name"]: config.paths.checkpoints_dir(r["name"]) for r in runs}
    mine: dict[str, subprocess.Popen] = {}
    attempts: dict[str, int] = {}
    failed: set[str] = set()

    while True:
        for name, process in list(mine.items()):
            if process.poll() is not None:
                status = process.returncode
                print(f"{name}: exit {status} {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                if status != 0 and not is_finished(folders[name]):
                    attempts[name] = attempts.get(name, 0) + 1
                    if attempts[name] >= MAX_ATTEMPTS:
                        failed.add(name)
                    else:
                        print(f"{name}: will resume, attempt {attempts[name] + 1} of {MAX_ATTEMPTS}", flush=True)
                del mine[name]
        stalled = [n for n, f in folders.items() if is_active(f) and min(age_seconds(f / "progress.log"), age_seconds(f / "claim")) > STALL_SECONDS]
        if stalled:
            stall_file = config.paths.results_dir("training_queue") / "STALLED"
            stall_file.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} no progress for {STALL_SECONDS // 60} minutes: {stalled}", encoding="utf-8")
        pending = [r for r in runs if not is_finished(folders[r["name"]]) and r["name"] not in mine and r["name"] not in failed and not is_active(folders[r["name"]])]
        active = sum(1 for r in runs if is_active(folders[r["name"]])) + len([n for n in mine if not is_active(folders[n])])
        if not pending and not mine and not any(is_active(f) for f in folders.values()):
            break
        if pending and active < args.max_active:
            run = pending[0]
            folder = folders[run["name"]]
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(script), run["config"], "--name", run["name"], "--override", *queue.get("common_overrides", []), *run["overrides"]]
            print(f"{run['name']}: start {time.strftime('%Y-%m-%d %H:%M:%S')} ({active + 1} active)", flush=True)
            mine[run["name"]] = subprocess.Popen(command)
            (folder / "claim").write_text(str(mine[run["name"]].pid), encoding="utf-8")
            time.sleep(20)
            continue
        time.sleep(30)
    print(f"queue complete, failed: {sorted(failed) or 'none'}", flush=True)


if __name__ == "__main__":
    main()
