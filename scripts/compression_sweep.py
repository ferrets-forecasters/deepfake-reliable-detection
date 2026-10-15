from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from score_clip_baseline import own_test  # noqa: E402

from deepfake_detection.config import load_config  # noqa: E402
from deepfake_detection.data.face_crop import FaceLandmarkDetector  # noqa: E402
from deepfake_detection.inference.clip_detector import ClipDetector, face_crops  # noqa: E402
from deepfake_detection.inference.factory import build_pipeline  # noqa: E402
from deepfake_detection.utils.watchdog import hang_watchdog  # noqa: E402

FFMPEG = Path(r"C:\tools\ffmpeg\bin\ffmpeg.exe")
ORIGINAL = "original"
BACKENDS = ("paper_model", "clip_ln_tuning")
COLUMNS = ["video_id", "dataset", "label", "condition", "p_fake", "units", "seconds", "error"]
MANIFEST_COLUMNS = ["video_id", "dataset", "label", "condition", "path", "bytes"]
UNIT_TIMEOUT_S = 300


def select_videos(config, datasets: list[str], per_class: int, seed: int) -> list[dict]:
    """Per dataset, the same number of real and fake own-test videos (at most `per_class` each), drawn with `seed`."""
    rng = random.Random(seed)
    chosen = []
    for dataset in datasets:
        videos = own_test(config, dataset)
        real = [v for v in videos if v["label"] == 0]
        fake = [v for v in videos if v["label"] == 1]
        n = min(per_class, len(real), len(fake))
        chosen += sorted(rng.sample(real, n) + rng.sample(fake, n), key=lambda v: v["video_id"])
    return chosen


def encode(source: Path, target: Path, crf: int) -> None:
    """H.264 at `crf`, same resolution, audio stream copied; falls back to AAC when the source audio cannot go into mp4 as is."""
    part = target.with_suffix(".part.mp4")
    base = [str(FFMPEG), "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
    try:
        subprocess.run(base + ["-c:a", "copy", str(part)], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(base + ["-c:a", "aac", "-b:a", "128k", str(part)], check=True)
    part.replace(target)


def build_conditions(videos: list[dict], crfs: list[int], root: Path, workers: int) -> list[dict]:
    jobs, rows = [], []
    for v in videos:
        rows.append({"video_id": f"{v['dataset']}__{v['video_id']}", "dataset": v["dataset"], "label": v["label"], "condition": ORIGINAL, "path": v["path"]})
        for crf in crfs:
            target = root / "raw" / f"crf{crf}" / f"{v['dataset']}__{v['video_id']}.mp4"
            rows.append({"video_id": f"{v['dataset']}__{v['video_id']}", "dataset": v["dataset"], "label": v["label"], "condition": f"crf{crf}", "path": target})
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((v["path"], target, crf))
    with ThreadPoolExecutor(workers) as pool:
        list(tqdm(pool.map(lambda j: encode(*j), jobs), total=len(jobs), desc="re-encode", unit="clip"))
    for r in rows:
        r["bytes"] = r["path"].stat().st_size
    return rows


def write_manifest(rows: list[dict], root: Path) -> None:
    (root / "labels").mkdir(parents=True, exist_ok=True)
    with open(root / "labels" / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows({**r, "path": str(r["path"])} for r in rows)


def read_manifest(path: Path) -> list[dict]:
    """Rows of a manifest; a relative clip path is read from the folder that holds `labels/`, so a manifest keeps working on another machine."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{**r, "label": int(r["label"]), "bytes": int(r["bytes"]), "path": Path(r["path"]) if Path(r["path"]).is_absolute() else path.parent.parent / r["path"]} for r in rows]


class PaperModel:
    def __init__(self, config, model: str):
        self.pipeline = build_pipeline(config, model, diarizer="single")

    def __call__(self, path: Path) -> tuple[float | None, int]:
        result = self.pipeline.run(path)
        return (result.video_p_fake if result.subjects else None), len(result.subjects)


class ClipLnTuning:
    def __init__(self, weights: Path, frames: int):
        self.detector, self.faces, self.frames = ClipDetector(weights), FaceLandmarkDetector(max_faces=5, margin=0.25), frames

    def __call__(self, path: Path) -> tuple[float | None, int]:
        crops, _ = face_crops(path, self.faces, self.frames)
        return (float(np.mean(self.detector.p_fake(crops))) if crops else None), len(crops)


def read_done(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.exists():
        return set()
    with open(csv_path, encoding="utf-8", newline="") as f:
        return {(r["video_id"], r["condition"]) for r in csv.DictReader(f)}


def score_backends(config, args, rows: list[dict], log) -> None:
    started, finished, total = time.time(), 0, len(rows) * len(args.backends)
    for backend in args.backends:
        out_dir = config.paths.predictions_dir(f"{args.model if backend == 'paper_model' else backend}_on_{args.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "predictions.csv"
        done = read_done(csv_path)
        finished += len(done)
        todo = [r for r in rows if (r["video_id"], r["condition"]) not in done]
        if not todo:
            continue
        scorer = PaperModel(config, args.model) if backend == "paper_model" else ClipLnTuning(args.weights, args.frames)
        new_file = not csv_path.exists()
        begin, base = time.time(), finished
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            for row in tqdm(todo, desc=backend, unit="video"):
                t0, p_fake, units, error = time.time(), "", 0, ""
                try:
                    with hang_watchdog(UNIT_TIMEOUT_S, log, f"{backend} {row['video_id']} {row['condition']}"):
                        value, units = scorer(row["path"])
                    p_fake = "" if value is None else round(float(value), 6)
                except Exception as exc:  # one broken clip must not stop the sweep
                    error = repr(exc)[:200]
                writer.writerow({"video_id": row["video_id"], "dataset": row["dataset"], "label": row["label"], "condition": row["condition"],
                                 "p_fake": p_fake, "units": units, "seconds": round(time.time() - t0, 2), "error": error})
                f.flush()
                finished += 1
                if (finished - base) % 10 == 0 or finished == total:
                    rate = (time.time() - begin) / (finished - base)
                    log.write(f"{time.strftime('%T')} {backend} {finished}/{total} elapsed {time.time() - started:.0f} s, {rate:.1f} s per video, "
                              f"this backend ETA {rate * (base + len(todo) - finished):.0f} s\n")
                    log.flush()


def metrics(labels, scores) -> dict:
    y, s = np.asarray(labels), np.asarray(scores)
    out = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    if len(set(y.tolist())) < 2:
        return out
    pred = (s >= 0.5).astype(int)
    return out | {"auc": round(float(roc_auc_score(y, s)), 4), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
                  "real_flagged": round(float(pred[y == 0].mean()), 4), "fake_caught": round(float(pred[y == 1].mean()), 4),
                  "mean_p_real": round(float(s[y == 0].mean()), 4), "mean_p_fake": round(float(s[y == 1].mean()), 4)}


def bootstrap_auc(y, scores_a, scores_b, n_boot: int, seed: int) -> dict:
    """95% interval of the AUC of `scores_a` and of the AUC of `scores_a` minus `scores_b` (or none), resampling videos."""
    rng = np.random.default_rng(seed)
    y, a = np.asarray(y), np.asarray(scores_a)
    b = None if scores_b is None else np.asarray(scores_b)
    aucs, deltas = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        auc_a = roc_auc_score(y[idx], a[idx])
        aucs.append(auc_a)
        if b is not None:
            deltas.append(auc_a - roc_auc_score(y[idx], b[idx]))
    out = {"auc_ci95": [round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)]}
    if deltas:
        out["auc_change_vs_original"] = round(float(np.mean(deltas)), 4)
        out["auc_change_ci95"] = [round(float(np.percentile(deltas, 2.5)), 4), round(float(np.percentile(deltas, 97.5)), 4)]
    return out


def summarize(config, args, manifest: list[dict], out_dir: Path) -> dict:
    sizes = {}
    for r in manifest:
        sizes.setdefault((r["dataset"], r["condition"]), []).append(r["bytes"])
    original_bytes = {r["video_id"]: r["bytes"] for r in manifest if r["condition"] == ORIGINAL}
    summary = {"videos": {d: {"real": sum(1 for r in manifest if r["dataset"] == d and r["condition"] == ORIGINAL and r["label"] == 0),
                              "fake": sum(1 for r in manifest if r["dataset"] == d and r["condition"] == ORIGINAL and r["label"] == 1)} for d in args.datasets},
               "size_ratio_to_original": {}, "backends": {}}
    for condition in [c for c in dict.fromkeys(r["condition"] for r in manifest)]:
        ratios = [r["bytes"] / original_bytes[r["video_id"]] for r in manifest if r["condition"] == condition]
        summary["size_ratio_to_original"][condition] = round(float(np.mean(ratios)), 3)
    for backend in args.backends:
        name = f"{args.model if backend == 'paper_model' else backend}_on_{args.name}"
        with open(config.paths.predictions_dir(name) / "predictions.csv", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["p_fake"] != ""]
        scores = {(r["video_id"], r["condition"]): (r["dataset"], int(r["label"]), float(r["p_fake"])) for r in rows}
        per_backend = {}
        for condition in dict.fromkeys(r["condition"] for r in manifest):
            per_backend[condition] = {}
            for dataset in [*args.datasets, "all"]:
                keys = [k for k, v in scores.items() if k[1] == condition and dataset in (v[0], "all")]
                if not keys:
                    continue
                entry = metrics([scores[k][1] for k in keys], [scores[k][2] for k in keys])
                if "auc" in entry:
                    paired = [k for k in keys if (k[0], ORIGINAL) in scores]
                    y = [scores[k][1] for k in paired]
                    base = None if condition == ORIGINAL else [scores[(k[0], ORIGINAL)][2] for k in paired]
                    entry |= bootstrap_auc(y, [scores[k][2] for k in paired], base, args.bootstrap, args.seed)
                per_backend[condition][dataset] = entry
        summary["backends"][backend] = per_backend
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return summary


def print_table(summary: dict) -> None:
    print(f"{'backend':15} {'condition':9} {'dataset':12} {'n':>4} {'AUC':>7} {'macroF1':>8} {'real flagged':>13} {'mean p real':>12}")
    for backend, conditions in summary["backends"].items():
        for condition, datasets in conditions.items():
            for dataset, m in datasets.items():
                if "auc" in m:
                    print(f"{backend:15} {condition:9} {dataset:12} {m['n']:>4} {m['auc']:>7.4f} {m['macro_f1']:>8.4f} {m['real_flagged']:>13.3f} {m['mean_p_real']:>12.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression sweep: the same own-test videos re-encoded with H.264 at several CRF levels, scored by the paper model and the CLIP LN-tuning detector. Resumable; one row per video, condition and backend.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--name", default="compression_sweep")
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--datasets", nargs="+", default=["hidf", "fakeavceleb"])
    parser.add_argument("--per-class", type=int, default=60)
    parser.add_argument("--crf", nargs="+", type=int, default=[23, 28, 35])
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=3, help="parallel ffmpeg processes")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-only", action="store_true", help="skip scoring and summarize the predictions that exist")
    parser.add_argument("--manifest", type=Path, default=None, help="score the clips of this manifest (written by this script or by pack_sweep_subset.py) instead of selecting and encoding videos")
    args = parser.parse_args()

    config = load_config(args.config)
    args.weights = args.weights or config.paths.datasets_root.parent / "external" / "sota" / "deepfake-detection" / "weights" / "model.torchscript"
    root = config.paths.datasets_root / args.name
    out_dir = config.paths.results_dir(args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        rows = read_manifest(args.manifest)
    else:
        rows = build_conditions(select_videos(config, args.datasets, args.per_class, args.seed), args.crf, root, args.workers)
        write_manifest(rows, root)
    with open(out_dir / "progress.log", "a", encoding="utf-8") as log:
        log.write(f"{time.strftime('%F %T')} {len({r['video_id'] for r in rows})} videos, {len(rows)} clips, backends {args.backends}\n")
        if args.summary_only:
            args.backends = [b for b in args.backends if (config.paths.predictions_dir(f"{args.model if b == 'paper_model' else b}_on_{args.name}") / "predictions.csv").exists()]
        else:
            score_backends(config, args, rows, log)
    print_table(summarize(config, args, rows, out_dir))


if __name__ == "__main__":
    main()
