from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.face_crop import FaceLandmarkDetector
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, SegmentEmbedder
from deepfake_detection.data.tiling import index_clips
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.reference import load_reference_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Control for scripts/score_images.py: the middle frame of FakeAVCeleb clips (a dataset the model was trained on) goes through the same still-image path, with no audio.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--per-kind", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    index = index_clips(root)
    rng = random.Random(args.seed)
    clips = {kind: sorted(p for (k, _, _), paths in index.items() if k == kind for p in paths) for kind in ("real", "fake_av", "fake_v")}
    chosen = [(path, int(kind != "real"), kind) for kind, paths in clips.items() for path in rng.sample(paths, min(args.per_kind, len(paths)))]

    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    scorer = CheckpointScorer(load_reference_repo(_project_root() / repro["reference_repo"]), config.paths.checkpoints_dir(args.model))
    embedder = SegmentEmbedder()
    detector = FaceLandmarkDetector(max_faces=1, margin=0.25)

    scores, labels, kinds = [], [], []
    for path, label, kind in tqdm(chosen, desc="clips"):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = detector.detect(frame)
        if not faces:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in faces[0].box)
        feature = embedder.video_features([[frame[y0:y1, x0:x1]] * VIDEOMAE_FRAMES])
        scores.append(scorer.score_visual_only(feature)[0])
        labels.append(label)
        kinds.append(kind)

    scores, labels, kinds = np.array(scores), np.array(labels), np.array(kinds)
    print(f"clips scored {len(scores)} of {len(chosen)}")
    for kind in ("real", "fake_av", "fake_v"):
        print(f"{kind}: n {int((kinds == kind).sum())}, mean p_fake {scores[kinds == kind].mean():.3f}")
    print(f"fake against real, still middle frame, no audio: AUC {roc_auc_score(labels, scores):.4f}")


if __name__ == "__main__":
    main()
