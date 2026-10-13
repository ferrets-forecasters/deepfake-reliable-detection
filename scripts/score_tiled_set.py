from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import FaceBoxTracker, MultiFaceTracker
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, WHISPER_SAMPLE_RATE, SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.reference import load_reference_repo

SEGMENT_SECONDS = 2.0
COLUMNS = [
    "sample_id", "label", "fake_side", "audio_from",
    "single_side", "single_p_fake", "single_alpha",
    "left_p_fake", "left_alpha", "right_p_fake", "right_alpha", "multi_p_fake",
]


def crops_for(reader, indices: list[int], boxes: dict) -> list[np.ndarray]:
    crops = []
    for idx in indices:
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[idx])
        crops.append(reader(idx)[y0:y1, x0:x1])
    return crops


def select_rows(rows: list[dict], per_plan: int) -> list[dict]:
    kept, seen = [], {}
    for row in rows:
        plan = (row["left_kind"], row["right_kind"], row["audio_from"])
        if seen.get(plan, 0) < per_plan:
            seen[plan] = seen.get(plan, 0) + 1
            kept.append(row)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the tiled two-person videos with an exported checkpoint: single-face baseline against one verdict per face track.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--dataset", default="fakeavceleb_tiled")
    parser.add_argument("--per-plan", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    scorer = CheckpointScorer(reference, config.paths.checkpoints_dir(args.model))
    embedder = SegmentEmbedder()
    fps_detect = config.preprocessing.video.face_detector_fps
    single_tracker, multi_tracker = FaceBoxTracker(fps_detect), MultiFaceTracker(fps_detect)

    root = config.paths.datasets_root / args.dataset
    with open(root / "labels" / "manifest.csv", encoding="utf-8") as f:
        rows = select_rows(list(csv.DictReader(f)), args.per_plan)

    out_dir = config.paths.predictions_dir(f"{args.model}_on_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.csv"
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {r["sample_id"] for r in csv.DictReader(f)}
    else:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    for row in tqdm(rows, desc=f"{args.model} on {args.dataset}"):
        if row["sample_id"] in done:
            continue
        path = root / row["raw_path"]
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps < SEGMENT_SECONDS:
            cap.release()
            continue
        indices = np.linspace(0, int(SEGMENT_SECONDS * fps) - 1, VIDEOMAE_FRAMES).round().astype(int).tolist()
        cache: dict[int, np.ndarray] = {}

        def reader(index: int) -> np.ndarray:
            if index not in cache:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                _, frame = cap.read()
                cache[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return cache[index]

        audio_feat = embedder.audio_feature(extract_audio_waveform(path, 0.0, SEGMENT_SECONDS, WHISPER_SAMPLE_RATE))

        def verdict(face_frames: list[np.ndarray]) -> tuple[float, float | None]:
            video_feat = embedder.video_feature(face_frames)
            return scorer.score(scorer.normalize(video_feat[None], audio_feat[None]))

        single_boxes = single_tracker.boxes_for_indices(reader, indices, fps)
        single_center = np.mean([(b[0] + b[2]) / 2 for b in single_boxes.values()]) / width
        single_p, single_alpha = verdict(crops_for(reader, indices, single_boxes))

        by_side: dict[str, tuple[float, float | None]] = {}
        for track in multi_tracker.tracks_for_indices(reader, indices, fps):
            center = np.mean([(b[0] + b[2]) / 2 for b in track.detections.values()]) / width
            by_side["left" if center < 0.5 else "right"] = verdict(crops_for(reader, indices, track.boxes_for_indices(indices)))
        cap.release()

        left, right = by_side.get("left", (np.nan, None)), by_side.get("right", (np.nan, None))
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow({
                "sample_id": row["sample_id"], "label": row["label"], "fake_side": row["fake_side"], "audio_from": row["audio_from"],
                "single_side": "left" if single_center < 0.5 else "right", "single_p_fake": round(single_p, 4), "single_alpha": single_alpha,
                "left_p_fake": round(left[0], 4), "left_alpha": left[1], "right_p_fake": round(right[0], 4), "right_alpha": right[1],
                "multi_p_fake": round(float(np.nanmax([left[0], right[0]])), 4),
            })


if __name__ == "__main__":
    main()
