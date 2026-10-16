from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from deepfake_detection.config import load_config

SCRIPTS = Path(__file__).parent


@dataclass
class Step:
    name: str
    command: list[str]
    marker: str | None     # file under the project root that exists when the step has finished; None: finished when the last run of the step was ok
    expected_seconds: int
    needs_idle_gpu: bool = False
    timeout_seconds: int = 0   # 0 means three times the expected time
    default: bool = True       # False: runs only when named with --only


STEPS = [
    Step("calibration_retrain", ["calibration.py", "--suffix", "_retrain"], "results/metrics/calibration_retrain/calibration.yaml", 180),
    Step("curves_retrain", ["prediction_curves.py", "--suffix", "_retrain"], "results/metrics/prediction_curves_retrain/metrics.yaml", 60),
    Step("speaker_splits", ["count_speaker_splits.py"], "results/analyses/single_voice_split_count/summary.yaml", 15),
    Step("dataset_releases", ["check_dataset_releases.py"], "results/metrics/dataset_releases/releases.yaml", 900),
    Step("benchmark_faces", ["benchmark_faces.py"], "results/metrics/benchmark_faces/summary.yaml", 420, needs_idle_gpu=True),
    Step("viral_cases_score", ["score_viral_cases.py", "--model", "moe_combined_retrain"], "results/analyses/viral_cases/summary.yaml", 900, default=False),
]


def fmt(seconds: float) -> str:
    return f"{int(seconds // 3600):d}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Runs the remaining experiment checks one after another: calibration and curves of the re-run leave-one-out models, the speaker split count, the dataset release facts and the cost per extra face. Resumable (a finished step is skipped), every step logs to its own file, the ETA is the expected time of the steps still to run, corrected by the durations measured so far.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", nargs="+", choices=[s.name for s in STEPS], help="run only these steps")
    parser.add_argument("--force", action="store_true", help="run the selected steps again even when they have finished")
    parser.add_argument("--list", action="store_true", help="print the steps and their state, run nothing")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.paths.datasets_root.parent
    work = root / "temp" / "run_remaining_checks"
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    selected = [s for s in STEPS if (s.name in args.only if args.only else s.default)]

    def finished(step: Step) -> bool:
        if step.marker is None:
            return state.get(step.name, {}).get("last_status") == "ok"
        return (root / step.marker).exists()

    if args.list:
        for s in STEPS:
            print(f"{s.name:22s} {'done' if finished(s) else 'to run':7s} expected {fmt(s.expected_seconds)}  {state.get(s.name, {}).get('last_status', '-')}")
        return

    todo = [s for s in selected if args.force or not finished(s)]
    log = open(work / "run.log", "a", encoding="utf-8")

    def say(message: str) -> None:
        line = f"{time.strftime('%F %T')} {message}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"start: {len(todo)} of {len(selected)} steps to run, expected {fmt(sum(s.expected_seconds for s in todo))}")
    measured_ratio, failures = 1.0, []
    for number, step in enumerate(todo, 1):
        remaining = sum(s.expected_seconds for s in todo[number - 1:]) * measured_ratio
        say(f"step {number}/{len(todo)} {step.name}: ETA for all remaining steps {fmt(remaining)}")
        step_log = work / f"{step.name}.log"
        start = time.time()
        limit = step.timeout_seconds or 3 * step.expected_seconds
        with open(step_log, "w", encoding="utf-8") as out:
            try:
                code = subprocess.run([sys.executable, str(SCRIPTS / step.command[0]), *step.command[1:]], stdout=out, stderr=subprocess.STDOUT, cwd=SCRIPTS.parent,
                                      timeout=limit).returncode
            except subprocess.TimeoutExpired:
                code = f"timeout after {fmt(limit)}"
        seconds = time.time() - start
        ok = code == 0 and (step.marker is None or finished(step))
        state[step.name] = {"last_status": "ok" if ok else f"failed (exit {code})", "seconds": round(seconds, 1), "finished_at": time.strftime("%F %T"), "log": str(step_log)}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        if ok:
            measured_ratio = max(0.2, min(5.0, seconds / step.expected_seconds))
        else:
            failures.append(step.name)
        say(f"step {step.name}: {'ok' if ok else 'FAILED, see ' + str(step_log)} in {fmt(seconds)}")

    say(f"finished, failed steps: {failures or 'none'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
