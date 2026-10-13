from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.evaluation.reproduce import test_records
from deepfake_detection.reference import load_reference_repo
from deepfake_detection.utils.progress import ResumableCsv, progress

VIDEO_DIM = 768
CONDITIONS = ["baseline", "audio_zeroed", "video_zeroed", "audio_from_other_video", "audio_shifted_one_segment", "audio_order_shuffled"]
COLUMNS = ["video", "label", "kind", "audio_avail"] + [f"{prefix}_{c}" for c in CONDITIONS for prefix in ("p", "a")]
CHUNK = 64


def load_features(files: list) -> tuple[np.ndarray, np.ndarray]:
    video, audio = [], []
    for path in files:
        # the embedding files are written by this project's own extraction, so unpickling them is safe
        with np.load(path, allow_pickle=True) as data:
            video.append(data["video_feat"].astype(np.float32))
            audio.append(data["audio_feat"].astype(np.float32))
    return np.stack(video), np.stack(audio)


def resample_in_time(features: np.ndarray, length: int) -> np.ndarray:
    return features[np.linspace(0, len(features) - 1, length).round().astype(int)]


def perturb(condition: str, audio: np.ndarray, other_audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if condition == "audio_from_other_video":
        return resample_in_time(other_audio, len(audio))
    if condition == "audio_shifted_one_segment" and len(audio) > 1:
        return np.roll(audio, 1, axis=0)
    if condition == "audio_order_shuffled" and len(audio) > 1:
        return audio[rng.permutation(len(audio))]
    return audio


def score_video(scorer: CheckpointScorer, index: int, features, partner_audio: np.ndarray, audio_avail: bool, seed: int) -> dict:
    video, audio = features
    rng = np.random.default_rng([seed, index])
    row = {}
    for condition in CONDITIONS:
        x = scorer.normalize(video, perturb(condition, audio, partner_audio, rng))
        if condition == "audio_zeroed" or not audio_avail:
            x[:, VIDEO_DIM:] = 0.0
        elif condition == "video_zeroed":
            x[:, :VIDEO_DIM] = 0.0
        p, alpha = scorer.score(x, audio_available=audio_avail)
        row[f"p_{condition}"], row[f"a_{condition}"] = round(p, 5), "" if alpha is None else round(alpha, 5)
    return row


def summarize(rows: list[dict], model: str, out_dir: Path) -> None:
    labels = np.array([int(r["label"]) for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    available = np.array([r["audio_avail"] in ("True", "1", True) for r in rows])
    report, shown = {}, lambda v: "-" if v is None else f"{v:.3f}"
    for scope, scope_mask in (("all test videos", np.ones(len(rows), dtype=bool)), ("datasets with audio", available)):
        print(f"\n{scope} ({int(scope_mask.sum())} videos)")
        print("condition | macro-F1 | AP | AUC | mean alpha | alpha audio-video fakes | alpha video-only fakes | alpha real")
        report[scope] = {}
        for condition in CONDITIONS:
            p = np.array([float(r[f"p_{condition}"]) for r in rows])[scope_mask]
            a = np.array([float(r[f"a_{condition}"]) if r[f"a_{condition}"] != "" else np.nan for r in rows])[scope_mask]
            y, k = labels[scope_mask], kinds[scope_mask]
            row = {"f1_macro": float(f1_score(y, p >= 0.5, average="macro")), "ap": float(average_precision_score(y, p)),
                   "auc": float(roc_auc_score(y, p)), "alpha_mean": float(np.nanmean(a))}
            for kind in ("audio_video", "video_only", "real"):
                mask = k == kind
                row[f"alpha_{kind}"] = float(np.nanmean(a[mask])) if mask.any() else None
            report[scope][condition] = {key: (round(v, 4) if v is not None else None) for key, v in row.items()}
            print(f"{condition} | {row['f1_macro']:.4f} | {row['ap']:.4f} | {row['auc']:.4f} | {row['alpha_mean']:.3f} | "
                  f"{shown(row['alpha_audio_video'])} | {shown(row['alpha_video_only'])} | {shown(row['alpha_real'])}")
    (out_dir / "metrics.yaml").write_text(yaml.safe_dump({"model": model, "videos": len(rows), "scopes": report}, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="What does each input stream do? Score the saved test embeddings with one stream removed, the audio taken from another video, shifted by one segment or shuffled. Resumable: finished videos are kept in records.csv and progress goes to progress.log.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="subject split seed of the model's training, when it is not the one in reproduction.yaml (seed runs 43 and 44)")
    parser.add_argument("--limit", type=int, default=0, help="first N test videos, 0 for all")
    parser.add_argument("--workers", type=int, default=6, help="threads that read embedding files")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.results_dir(f"expert_tests_{args.model}")
    state = ResumableCsv(out_dir / "records.csv", COLUMNS, key="video")
    if args.summarize_only:
        summarize(state.rows, args.model, out_dir)
        return

    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    scorer = CheckpointScorer(reference, config.paths.checkpoints_dir(args.model))
    split = {**repro["split"], **({"seed": args.split_seed} if args.split_seed is not None else {})}
    records = test_records(reference, config, repro["models"].get(args.model, {"experiment": "combined"}), split)
    if args.limit:
        records = records[: args.limit]
    todo = [i for i, r in enumerate(records) if str(r[0][0]) not in state.done]

    bar = progress(len(records), len(records) - len(todo), out_dir / "progress.log", f"{args.model} expert tests")
    with ThreadPoolExecutor(args.workers) as pool:
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start:start + CHUNK]
            needed = sorted(set(chunk) | {(i - 1) % len(records) for i in chunk})
            loaded = dict(zip(needed, pool.map(lambda i: _safe_load(records[i][0]), needed)))
            for i in chunk:
                files, label, _, kind, audio_avail = records[i]
                features, partner = loaded[i], loaded[(i - 1) % len(records)]
                if features is None or partner is None:
                    bar.update(1)
                    continue
                row = {"video": str(files[0]), "label": int(label), "kind": kind, "audio_avail": bool(audio_avail)}
                row.update(score_video(scorer, i, features, partner[1], bool(audio_avail), args.seed))
                state.append(row)
                bar.update(1)
    bar.close()
    summarize(state.rows, args.model, out_dir)


def _safe_load(files: list):
    try:
        return load_features(files)
    except Exception:
        return None


if __name__ == "__main__":
    main()
