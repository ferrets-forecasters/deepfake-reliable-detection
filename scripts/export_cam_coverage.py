from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, WHISPER_SAMPLE_RATE, SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference.multi_subject import crop_track_frames, frame_reader, segment_frame_range
from deepfake_detection.reference import load_reference_repo
from deepfake_detection.utils.gpu import cap_gpu_memory
from deepfake_detection.xai.cross_modal import video_cam_for_segment
from deepfake_detection.xai.overlay import box_to_crop

SEGMENT_SECONDS = 2.0
CROP_MARGIN = 0.25
MANIFEST = ["id", "dataset", "label", "manip_type", "path", "p_fake", "alpha", "why_chosen"]
COVERAGE = ["id", "dataset", "label", "segment", "start_s", "p_fake_segment", "alpha_segment", "mass_in_face_box", "mass_outside_face_box",
            "mass_in_mouth", "uniform_face_share", "uniform_mouth_share", "peak_in_face_box", "peak_in_mouth"]


def overlap_weights(box: tuple[float, float, float, float], grid: int) -> np.ndarray:
    """Fraction of every grid cell that lies inside a box given as fractions of the crop."""
    edges = np.linspace(0.0, 1.0, grid + 1)
    x = np.clip(np.minimum(edges[1:], box[2]) - np.maximum(edges[:-1], box[0]), 0, None) * grid
    y = np.clip(np.minimum(edges[1:], box[3]) - np.maximum(edges[:-1], box[1]), 0, None) * grid
    return np.outer(y, x)


def tight_face_box() -> tuple[float, float, float, float]:
    """The face box inside the crop: the crop is the landmark extent grown by CROP_MARGIN on every side."""
    low = CROP_MARGIN / (1 + 2 * CROP_MARGIN)
    return low, low, 1 - low, 1 - low


def coverage(cam: np.ndarray, mouth_fractions: list[tuple[float, float, float, float]]) -> dict:
    positive = np.maximum(cam, 0.0)
    steps, grid = positive.shape[0], positive.shape[-1]
    total = positive.sum()
    face_w = overlap_weights(tight_face_box(), grid)
    mouth_w = [overlap_weights(box, grid) for box in mouth_fractions[:steps]]
    inside_face = float((positive * face_w).sum() / total) if total > 0 else float("nan")
    inside_mouth = float(sum((positive[i] * w).sum() for i, w in enumerate(mouth_w)) / total) if total > 0 else float("nan")
    peak = np.unravel_index(positive.argmax(), positive.shape)
    return {
        "mass_in_face_box": inside_face, "mass_outside_face_box": 1 - inside_face, "mass_in_mouth": inside_mouth,
        "uniform_face_share": float(face_w.mean()), "uniform_mouth_share": float(np.mean([w.mean() for w in mouth_w])),
        "peak_in_face_box": bool(face_w[peak[1], peak[2]] >= 0.5), "peak_in_mouth": bool(mouth_w[peak[0]][peak[1], peak[2]] >= 0.5),
    }


NO_AUDIO = {"celebdfv2", "ffpp"}
FFPP_METHODS = {"deepfakes", "face2face", "faceshifter", "faceswap", "neuraltextures"}


def score_key(path: Path) -> str:
    """The sample id of a video in the saved predictions: FF++ fakes carry their method as a prefix."""
    method = path.parent.name.lower()
    return f"{method}_{path.stem}" if method in FFPP_METHODS else path.stem


def test_videos(config, dataset: str) -> list[tuple[Path, int, str]]:
    if dataset in NO_AUDIO:
        manifest = config.paths.analysis_dir("example_pairs") / dataset / "manifest.csv"
        with open(manifest, encoding="utf-8", newline="") as f:
            return [(Path(r["video"]), int(r["label"]), r["manip_type"]) for r in csv.DictReader(f)]
    root = config.paths.datasets_root / dataset
    with open(root / "labels" / "test_split.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["own_test_seed42"] == "1"]
    if dataset == "hidf":
        base = root / "raw" / "test_split"
        return [(base / r["member"], int(r["label"]), r["manip_type"]) for r in rows]
    base = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    out = []
    for r in rows:
        member = Path(*Path(r["member"]).parts[-5:])
        if member.parts[0] in ("RealVideo-RealAudio", "FakeVideo-FakeAudio"):
            out.append((base / member, int(r["label"]), "real" if r["label"] == "0" else "fake_video_fake_audio"))
    return out


def predictions(config, model_scope: str) -> dict[str, tuple[float, float]]:
    path = config.paths.predictions_dir(model_scope) / "predictions.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[re.sub(r"_seg\d+$", "", Path(row["video"]).stem)] = (float(row["p_fake_best_f1"]), float(row["alpha_best_f1"]))
    return out


def draw(videos, scores, per_label: int, failures: int, seed: int) -> list[tuple[Path, int, str, str]]:
    rng = random.Random(seed)
    chosen = []
    for label in (0, 1):
        pool = [v for v in videos if v[1] == label and v[0].exists()]
        rng.shuffle(pool)
        wrong = [v for v in pool if score_key(v[0]) in scores and ((label == 1 and scores[score_key(v[0])][0] < 0.5) or (label == 0 and scores[score_key(v[0])][0] >= 0.5))]
        right = [v for v in pool if v not in wrong]
        picks = [(*v, "failure: fake missed or real flagged at 0.5") for v in wrong[:failures]]
        picks += [(*v, "seeded random draw from the test split") for v in right[:per_label - len(picks)]]
        chosen += picks
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual activation maps for seeded test-split videos: the map of up to K segments per video with its face crop, and the share of the map inside the face box, inside the mouth and outside the box.")
    parser.add_argument("--dataset", required=True, choices=["fakeavceleb", "hidf", "celebdfv2", "ffpp"])
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--per-label", type=int, default=4)
    parser.add_argument("--failures", type=int, default=1)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--maps-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    args = parser.parse_args()

    cap_gpu_memory(args.gpu_memory_fraction)
    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    scorer = CheckpointScorer(load_reference_repo(_project_root() / repro["reference_repo"]), config.paths.checkpoints_dir(args.model))
    embedder = SegmentEmbedder()
    tracker = MultiFaceTracker(config.preprocessing.video.face_detector_fps, max_faces=1)

    scores = predictions(config, f"{args.model}_on_combined_test_seed42")
    chosen = draw(test_videos(config, args.dataset), scores, args.per_label, args.failures, args.seed)
    maps_dir = args.maps_dir or config.paths.xai_dir(f"activation_maps_{args.model}") / args.dataset
    out_dir = args.out or config.paths.analysis_dir("cam_coverage") / args.dataset
    for folder in (out_dir, maps_dir if args.save_maps else out_dir):
        folder.mkdir(parents=True, exist_ok=True)

    manifest, rows = [], []
    for number, (path, label, manip, why) in enumerate(tqdm(chosen, desc="videos")):
        video_id = f"{args.dataset}_{'fake' if label else 'real'}_{number:03d}"
        video = cv2.VideoCapture(str(path))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        n_segments = int((frame_count / fps) // SEGMENT_SECONDS)
        if n_segments == 0:
            video.release()
            continue
        reader = frame_reader(video, frame_count)
        try:
            tracks = tracker.tracks_for_indices(reader, [0, frame_count - 1], fps)
        except ValueError:
            video.release()
            continue
        track = tracks[0]
        usable = [s for s in range(n_segments) if track.present_between(*segment_frame_range(s, SEGMENT_SECONDS, fps), int(0.6 * fps))]
        picks = sorted({usable[i] for i in np.linspace(0, len(usable) - 1, min(args.segments, len(usable))).round().astype(int)}) if usable else []
        for segment in picks:
            first, last = segment_frame_range(segment, SEGMENT_SECONDS, fps)
            indices = np.linspace(first, last, VIDEOMAE_FRAMES).round().astype(int).tolist()
            boxes = track.boxes_for_indices(indices)
            mouths = track.mouth_boxes_for_indices(indices)
            crops = crop_track_frames(reader, indices, boxes)
            has_audio = args.dataset not in NO_AUDIO
            if has_audio:
                waveform = extract_audio_waveform(path, segment * SEGMENT_SECONDS, (segment + 1) * SEGMENT_SECONDS, WHISPER_SAMPLE_RATE)
                audio_feat = embedder.audio_feature(waveform)
            else:
                audio_feat = np.zeros(1024, dtype=np.float32)
            cam = video_cam_for_segment(embedder, scorer, crops, audio_feat, has_audio)
            video_feat = embedder.video_feature(crops)
            if has_audio:
                p_segment, alpha_segment = scorer.score(scorer.normalize(video_feat[None], audio_feat[None]))
            else:
                p_segment, alpha_segment = scorer.score_visual_only(video_feat[None])
            per_step = [tuple(np.mean([box_to_crop(mouths[indices[2 * s + k]], boxes[indices[2 * s + k]]) for k in (0, 1)], axis=0)) for s in range(len(indices) // 2)]
            stats = coverage(cam, per_step)
            rows.append({"id": video_id, "dataset": args.dataset, "label": label, "segment": segment, "start_s": segment * SEGMENT_SECONDS,
                         "p_fake_segment": round(p_segment, 4), "alpha_segment": "" if alpha_segment is None else round(alpha_segment, 4),
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}})
            if args.save_maps:
                step = int(np.maximum(cam, 0).reshape(len(cam), -1).sum(axis=1).argmax())
                shown = indices[min(2 * step, len(indices) - 1)]
                target = maps_dir / video_id
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target / f"segment_{segment:02d}.npz", frames=np.stack([cv2.resize(c, (224, 224)) for c in crops]), cam=cam, step=step,
                    p_fake=p_segment, alpha=np.nan if alpha_segment is None else alpha_segment, full_frame=reader(shown),
                    box=np.array(boxes[shown], dtype=np.float32), mouth_box=np.array(mouths[shown], dtype=np.float32),
                    face_box_in_crop=np.array(tight_face_box(), dtype=np.float32), mouth_boxes_in_crop=np.array(per_step, dtype=np.float32),
                )
        video.release()
        p, a = scores.get(score_key(path), ("", ""))
        manifest.append({"id": video_id, "dataset": args.dataset, "label": label, "manip_type": manip, "path": str(path), "p_fake": p, "alpha": a, "why_chosen": why})

    for name, columns, content in (("manifest.csv", MANIFEST, manifest), ("coverage.csv", COVERAGE, rows)):
        with open(out_dir / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(content)
    if rows:
        for label in (0, 1):
            part = [r for r in rows if r["label"] == label]
            if part:
                print(f"{args.dataset} {'fake' if label else 'real'}: {len(part)} segments, mean mass in face box {np.mean([r['mass_in_face_box'] for r in part]):.3f} "
                      f"(uniform {np.mean([r['uniform_face_share'] for r in part]):.3f}), in mouth {np.mean([r['mass_in_mouth'] for r in part]):.3f} "
                      f"(uniform {np.mean([r['uniform_mouth_share'] for r in part]):.3f})")
    print(f"{len(manifest)} videos, {len(rows)} segments in {out_dir}")


if __name__ == "__main__":
    main()
