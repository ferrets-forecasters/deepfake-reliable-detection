from __future__ import annotations

import argparse
import csv
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference import MultiSubjectInference, SingleSpeaker
from deepfake_detection.reference import load_reference_repo
from deepfake_detection.utils.watchdog import hang_watchdog

STAGES = ["detect_track", "diarize", "audio_embed", "video_embed", "score"]
COLUMNS = ["clip", "faces", "diarizer", "path", "duration_s", "tracks_found", "segments_scored", "total_s", *[f"{s}_s" for s in STAGES], "other_s"]


class Timer:
    def __init__(self):
        self.seconds = dict.fromkeys(STAGES, 0.0)

    def reset(self) -> None:
        self.seconds = dict.fromkeys(STAGES, 0.0)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(timer: Timer, stage: str, fn):
    def wrapper(*args, **kwargs):
        sync()
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            sync()
            timer.seconds[stage] += time.perf_counter() - start
    return wrapper


class TimedDiarizer:
    def __init__(self, inner, timer: Timer):
        self.inner, self.timer = inner, timer

    def __call__(self, *args, **kwargs):
        return timed(self.timer, "diarize", self.inner)(*args, **kwargs)


def gpu_busy(limit_percent: int) -> str | None:
    """Returns a message when the GPU is not idle: a timing taken next to another job is inflated."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    utilization, memory = (int(x) for x in out.strip().splitlines()[0].split(","))
    return f"GPU at {utilization}% with {memory} MiB in use" if utilization > limit_percent else None


def pairs(config, count: int, seed: int) -> list[dict]:
    """Tiled own-test videos (two faces) with the left source clip of each alone (one face)."""
    root = config.paths.datasets_root / "fakeavceleb_tiled_own_test_lossless"
    source_root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    with open(root / "labels" / "manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        tiled, single = root / r["raw_path"], source_root / Path(r["left_clip"].replace("\\", "/"))
        if tiled.exists() and single.exists():
            out.append({"clip": r["sample_id"], "two": tiled, "one": single})
        if len(out) == count:
            break
    return out


def build_pipeline(args, config, diarizer_name: str, timer: Timer) -> MultiSubjectInference:
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    tracker = MultiFaceTracker(config.preprocessing.video.face_detector_fps, max_faces=args.max_faces)
    embedder, scorer = SegmentEmbedder(), CheckpointScorer(reference, config.paths.checkpoints_dir(args.model))
    if diarizer_name == "ecapa":
        from deepfake_detection.inference.diarization import EcapaDiarizer

        diarizer = EcapaDiarizer()
    else:
        diarizer = SingleSpeaker()
    tracker.tracks_for_indices = timed(timer, "detect_track", tracker.tracks_for_indices)
    embedder.audio_features = timed(timer, "audio_embed", embedder.audio_features)
    embedder.video_features = timed(timer, "video_embed", embedder.video_features)
    scorer.score = timed(timer, "score", scorer.score)
    return MultiSubjectInference(tracker=tracker, embedder=embedder, scorer=scorer, diarizer=TimedDiarizer(diarizer, timer))


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for diarizer in sorted({r["diarizer"] for r in rows}):
        by_clip: dict[str, dict[int, dict]] = {}
        for r in rows:
            if r["diarizer"] == diarizer:
                by_clip.setdefault(r["clip"], {})[int(r["faces"])] = r
        complete = {c: v for c, v in by_clip.items() if 1 in v and 2 in v and int(v[2]["tracks_found"]) == 2 and int(v[1]["tracks_found"]) == 1}
        entry: dict = {"clips_with_one_and_two_faces_found": len(complete), "clips_run": len(by_clip)}
        for faces in (1, 2):
            sel = [v[faces] for v in complete.values()]
            entry[f"faces_{faces}"] = {k: [round(float(np.mean([float(r[k]) for r in sel])), 4), round(float(np.std([float(r[k]) for r in sel])), 4)]
                                       for k in ["total_s", *[f"{s}_s" for s in STAGES], "other_s"]} if sel else {}
        if complete:
            ratios = np.array([float(v[2]["total_s"]) / float(v[1]["total_s"]) for v in complete.values()])
            boot = np.random.default_rng(42).choice(ratios, size=(2000, len(ratios))).mean(axis=1)
            entry["time_ratio_two_over_one"] = {"mean": round(float(ratios.mean()), 3), "ci95": [round(float(np.quantile(boot, 0.025)), 3), round(float(np.quantile(boot, 0.975)), 3)]}
            for s in STAGES:
                one = np.mean([float(v[1][f"{s}_s"]) for v in complete.values()])
                two = np.mean([float(v[2][f"{s}_s"]) for v in complete.values()])
                entry.setdefault("stage_ratio_two_over_one", {})[s] = round(float(two / one), 3) if one > 0 else None
        out[diarizer] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost of an extra face: the full multi-person pipeline on a one-face clip and on the two-face tiled video that contains it, timed per stage on an idle GPU. Resumable, one row per run in timings.csv.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--clips", type=int, default=12, help="tiled videos, each timed once with two faces and once with one")
    parser.add_argument("--diarizers", nargs="+", default=["single", "ecapa"], choices=["single", "ecapa"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gpu-utilization", type=int, default=15)
    parser.add_argument("--allow-busy", action="store_true", help="time anyway while another job uses the GPU (the numbers are inflated)")
    args = parser.parse_args()

    busy = gpu_busy(args.max_gpu_utilization)
    if busy and not args.allow_busy:
        raise SystemExit(f"{busy}: stop the other job or pass --allow-busy")

    config = load_config(args.config)
    out_dir = config.paths.results_dir("benchmark_faces")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "timings.csv"
    done: set[tuple[str, int, str]] = set()
    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        done = {(r["clip"], int(r["faces"]), r["diarizer"]) for r in rows}

    items = pairs(config, args.clips, args.seed)
    if not items:
        raise SystemExit("no tiled own-test video and left source clip found on disk")
    todo = [(it, d) for d in args.diarizers for it in items if not all((it["clip"], f, d) in done for f in (1, 2))]
    log = open(out_dir / "progress.log", "a", encoding="utf-8")
    log.write(f"{time.strftime('%F %T')} start: {len(todo)} clip and diarizer pairs to time, {len(done)} runs already in timings.csv\n")

    timer = Timer()
    new_file = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for diarizer_name in args.diarizers:
            batch = [it for it, d in todo if d == diarizer_name]
            if not batch:
                continue
            pipeline = build_pipeline(args, config, diarizer_name, timer)
            for it in items[: args.warmup]:
                pipeline.run(it["two"])
            for it in tqdm(batch, desc=f"{diarizer_name} diarizer", unit="clip"):
                for faces, key in ((1, "one"), (2, "two")):
                    if (it["clip"], faces, diarizer_name) in done:
                        continue
                    timer.reset()
                    sync()
                    start = time.perf_counter()
                    with hang_watchdog(120, log, f"{diarizer_name} {it['clip']} faces {faces}"):
                        result = pipeline.run(it[key])
                    sync()
                    total = time.perf_counter() - start
                    row = {"clip": it["clip"], "faces": faces, "diarizer": diarizer_name, "path": str(it[key]), "duration_s": round(result.duration, 2),
                           "tracks_found": len(result.subjects), "segments_scored": sum(len(s.segments) for s in result.subjects), "total_s": round(total, 4)}
                    row.update({f"{s}_s": round(timer.seconds[s], 4) for s in STAGES})
                    row["other_s"] = round(total - sum(timer.seconds.values()), 4)
                    writer.writerow(row)
                    f.flush()
                    rows.append({k: str(v) for k, v in row.items()})
                    log.write(f"{time.strftime('%T')} {diarizer_name} {it['clip']} faces {faces} total {total:.2f} s tracks {row['tracks_found']}\n")
                    log.flush()

    summary = summarize(rows)
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
