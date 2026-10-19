from __future__ import annotations

import argparse
import random

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from deepfake_detection.config import load_config
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import SegmentEmbedder
from deepfake_detection.data.tiling import index_clips
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference import MultiSubjectInference
from deepfake_detection.reference import load_reference
from deepfake_detection.utils.progress import ResumableCsv, progress

KINDS = {"real": "real video, real audio", "fake_av": "fake video, fake audio", "fake_v": "fake video, real audio", "fake_a": "real video, fake audio"}
VIDEO_DIM = 768
COLUMNS = ["clip", "kind", "model", "p_fake", "alpha"]


class SeveralScorers:
    """Lets one pass of the pipeline score the same features with several checkpoints: the raw
    features go through each model's own normalisation and the results are kept per model."""

    def __init__(self, scorers: dict[str, CheckpointScorer]):
        self.scorers = scorers
        self.last: dict[str, tuple[float, float | None]] = {}

    def normalize(self, video: np.ndarray, audio: np.ndarray) -> np.ndarray:
        return np.concatenate([video, audio], axis=1)

    def score(self, x: np.ndarray) -> tuple[float, float | None]:
        for name, scorer in self.scorers.items():
            self.last[name] = scorer.score(scorer.normalize(x[:, :VIDEO_DIM], x[:, VIDEO_DIM:]))
        return next(iter(self.last.values()))


def summarize(rows: list[dict], models: list[str]) -> None:
    print("model | category | n | mean p_fake | flagged at 0.5 | mean alpha | AUC against real | AP against real")
    for name in models:
        real = [r for r in rows if r["model"] == name and r["kind"] == "real"]
        for kind, text in KINDS.items():
            group = [r for r in rows if r["model"] == name and r["kind"] == kind]
            if not group:
                continue
            p = np.array([float(r["p_fake"]) for r in group])
            a = np.array([float(r["alpha"]) for r in group if r["alpha"] not in ("", "None")], dtype=float)
            line = f"{name} | {text} | {len(group)} | {p.mean():.3f} | {(p >= 0.5).mean():.2f} | {a.mean() if len(a) else float('nan'):.3f}"
            if kind != "real" and real:
                y = np.r_[np.ones(len(group)), np.zeros(len(real))]
                s = np.r_[p, [float(r["p_fake"]) for r in real]]
                line += f" | {roc_auc_score(y, s):.3f} | {average_precision_score(y, s):.3f}"
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score FakeAVCeleb clips by manipulation category (which modality was faked) with several checkpoints. Resumable: finished clips are kept in clips.csv and progress goes to progress.log.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="+", default=["moe_leave_fakeavceleb", "moe_combined"])
    parser.add_argument("--name", default="fakeavceleb_categories", help="folder under results/inferences/ that holds clips.csv")
    parser.add_argument("--per-kind", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.predictions_dir(args.name)
    state = ResumableCsv(out_dir / "clips.csv", COLUMNS, key="clip")
    if args.summarize_only:
        summarize(state.rows, args.models)
        return

    reference = load_reference()
    scorers = SeveralScorers({name: CheckpointScorer(reference, config.paths.checkpoints_dir(name)) for name in args.models})
    pipeline = MultiSubjectInference(tracker=MultiFaceTracker(config.preprocessing.video.face_detector_fps, max_faces=1), embedder=SegmentEmbedder(), scorer=scorers)

    index = index_clips(config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2")
    rng = random.Random(args.seed)
    chosen = []
    for kind in KINDS:
        paths = sorted(p for (k, _, _), items in index.items() if k == kind for p in items)
        chosen += [(kind, path) for path in rng.sample(paths, min(args.per_kind, len(paths)))]
    todo = [(kind, path) for kind, path in chosen if str(path.relative_to(path.parents[4])) not in state.done]

    bar = progress(len(chosen), len(chosen) - len(todo), out_dir / "progress.log", "fakeavceleb categories")
    for kind, path in todo:
        result = pipeline.run(path)
        key = str(path.relative_to(path.parents[4]))
        if result.subjects:
            for name, (p_fake, alpha) in scorers.last.items():
                state.append({"clip": key, "kind": kind, "model": name, "p_fake": round(p_fake, 5), "alpha": "" if alpha is None else round(alpha, 5)})
        else:
            state.append({"clip": key, "kind": kind, "model": "none", "p_fake": "", "alpha": ""})
        bar.update(1)
    bar.close()
    summarize(state.rows, args.models)


if __name__ == "__main__":
    main()
