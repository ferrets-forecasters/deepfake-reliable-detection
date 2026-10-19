from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, WHISPER_SAMPLE_RATE, SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference.multi_subject import crop_track_frames, frame_reader, segment_frame_range
from deepfake_detection.reference import load_reference
from deepfake_detection.utils.gpu import cap_gpu_memory
from deepfake_detection.xai.lipsync import shift_waveform

SEGMENT_SECONDS = 2.0
VIDEO_DIM = 768
CATEGORIES = {"real": "RealVideo-RealAudio", "fake_video_fake_audio": "FakeVideo-FakeAudio", "fake_video_real_audio": "FakeVideo-RealAudio", "real_video_fake_audio": "RealVideo-FakeAudio"}
LAGS_MS = list(range(-400, 401, 40))
SHIFTS_MS = [-300, -200, -100, 0, 100, 200, 300]
LABEL = {"real": 0, "fake_video_fake_audio": 1, "fake_video_real_audio": 1, "real_video_fake_audio": 1}


def test_clips(config) -> dict[str, list[Path]]:
    base = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
    out: dict[str, list[Path]] = {k: [] for k in CATEGORIES}
    with open(config.paths.datasets_root / "fakeavceleb" / "labels" / "test_split.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["own_test_seed42"] != "1":
                continue
            member = Path(*Path(r["member"]).parts[-5:])
            for key, folder in CATEGORIES.items():
                if member.parts[0] == folder and (base / member).exists():
                    out[key].append(base / member)
    return out


def predictions(config) -> dict[str, tuple[float, float]]:
    path = config.paths.predictions_dir("moe_combined_retrain_on_combined_test_seed42") / "predictions.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[re.sub(r"_seg\d+$", "", Path(r["video"]).stem)] = (float(r["p_fake_best_f1"]), float(r["alpha_best_f1"]))
    return out


def envelope(waveform: np.ndarray, fps: float, count: int) -> np.ndarray:
    """RMS of the audio in every video frame interval."""
    edges = np.linspace(0, len(waveform), count + 1).astype(int)
    return np.array([np.sqrt(np.mean(waveform[a:b] ** 2)) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])


def lag_curve(motion: np.ndarray, audio: np.ndarray, fps: float) -> dict[int, float]:
    """Pearson correlation of lip motion with the audio envelope delayed by each lag (positive lag = audio later than the lips)."""
    out = {}
    n = min(len(motion), len(audio))
    motion, audio = motion[:n], audio[:n]
    for lag in LAGS_MS:
        k = int(round(lag / 1000 * fps))
        a = np.roll(audio, k)
        lo, hi = max(k, 0), n + min(k, 0)
        x, y = motion[lo:hi], a[lo:hi]
        out[lag] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 and x.std() > 1e-9 and y.std() > 1e-9 else float("nan")
    return out


class ClipFeatures:
    """Video vectors of one clip's segments, audio waveform and per-frame lip opening."""

    def __init__(self, path: Path, embedder, tracker):
        cap = cv2.VideoCapture(str(path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        reader = frame_reader(cap, count)
        tracks = tracker.tracks_for_indices(reader, [0, count - 1], self.fps)
        track = tracks[0]
        self.segments = int((count / self.fps) // SEGMENT_SECONDS)
        crops = []
        for s in range(self.segments):
            first, last = segment_frame_range(s, SEGMENT_SECONDS, self.fps)
            idx = np.linspace(first, last, VIDEOMAE_FRAMES).round().astype(int).tolist()
            crops.append(crop_track_frames(reader, idx, track.boxes_for_indices(idx)))
        self.video = embedder.video_features(crops)
        self.waveform = extract_audio_waveform(path, 0.0, self.segments * SEGMENT_SECONDS, WHISPER_SAMPLE_RATE)
        frames = np.arange(int(self.segments * SEGMENT_SECONDS * self.fps))
        opening = track.mouth_open_for_indices(frames.tolist())
        self.mouth = np.array([opening[i] for i in frames])
        cap.release()

    def audio_features(self, embedder, waveform: np.ndarray) -> np.ndarray:
        samples = int(SEGMENT_SECONDS * WHISPER_SAMPLE_RATE)
        return embedder.audio_features([waveform[i * samples:(i + 1) * samples] for i in range(self.segments)])


def score(scorer, video: np.ndarray, audio: np.ndarray | None, zero_video: bool = False) -> tuple[float, float]:
    x = scorer.normalize(video, audio if audio is not None else np.zeros((len(video), 1024), dtype=np.float32))
    if audio is None:
        x[:, VIDEO_DIM:] = 0.0
    if zero_video:
        x[:, :VIDEO_DIM] = 0.0
    p, a = scorer.score(x)
    return round(p, 5), (round(a, 5) if a is not None else float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio tests on seeded test-split FakeAVCeleb clips (inference only): lip-motion against audio-energy lag per category and P(fake) and alpha with the audio shifted, then P(fake) and alpha under audio removal, shifts, the audio of another video, the fake audio of the same source and removed video.")
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--per-category", type=int, default=25)
    parser.add_argument("--conditions-clips", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-fraction", type=float, default=None)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cap_gpu_memory(args.gpu_memory_fraction)
    config = load_config(args.config)
    scorer = CheckpointScorer(load_reference(), config.paths.checkpoints_dir(args.model))
    embedder = SegmentEmbedder()
    tracker = MultiFaceTracker(25, max_faces=1)
    clips = test_clips(config)
    scores = predictions(config)
    rng = random.Random(args.seed)
    for items in clips.values():
        items.sort()
        rng.shuffle(items)

    out_sync = config.paths.analysis_dir("av_sync_fakeavceleb")
    out_cond = config.paths.analysis_dir("alpha_conditions")
    out_sync.mkdir(parents=True, exist_ok=True)
    out_cond.mkdir(parents=True, exist_ok=True)

    lag_rows, shift_rows, manifest = [], [], []
    for category, items in clips.items():
        done = 0
        for path in tqdm(items, desc=category, total=args.per_category):
            if done >= args.per_category:
                break
            try:
                feats = ClipFeatures(path, embedder, tracker)
            except Exception:
                continue
            if feats.segments == 0:
                continue
            done += 1
            motion = np.abs(np.diff(feats.mouth, prepend=feats.mouth[0]))
            curve = lag_curve(motion, envelope(feats.waveform, feats.fps, len(feats.mouth)), feats.fps)
            best = max((k for k in curve if not np.isnan(curve[k])), key=lambda k: curve[k], default=None)
            clip_id = f"{category}_{done:02d}"
            for lag, r in curve.items():
                lag_rows.append({"id": clip_id, "category": category, "lag_ms": lag, "correlation": round(r, 4) if not np.isnan(r) else ""})
            for shift in SHIFTS_MS:
                p, a = score(scorer, feats.video, feats.audio_features(embedder, shift_waveform(feats.waveform, int(round(shift * WHISPER_SAMPLE_RATE / 1000)))))
                shift_rows.append({"id": clip_id, "category": category, "shift_ms": shift, "p_fake": p, "alpha": a})
            base = scores.get(path.stem, ("", ""))
            manifest.append({"id": clip_id, "category": category, "label": LABEL[category], "path": str(path), "p_fake": base[0], "alpha": base[1], "segments": feats.segments,
                             "best_lag_ms": best if best is not None else "", "best_correlation": round(curve[best], 4) if best is not None else "",
                             "correlation_at_0": round(curve[0], 4) if not np.isnan(curve[0]) else "", "why_chosen": "seeded random draw from the test split"})
    for name, rows in (("lag_curves.csv", lag_rows), ("shifted_audio.csv", shift_rows), ("manifest.csv", manifest)):
        with open(out_sync / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # alpha under conditions: real clips that have a fake-audio twin (same identity folder and clip number), plus fake clips
    fake_audio = {(p.parts[-2], p.stem.replace("_fake", "")): p for p in clips["real_video_fake_audio"]}
    pool = [p for p in clips["real"] if (p.parts[-2], p.stem) in fake_audio]
    chosen = [(p, "real") for p in pool[: args.conditions_clips * 2 // 3]] + [(p, "fake_video_fake_audio") for p in clips["fake_video_fake_audio"][: args.conditions_clips // 3]]
    other_pool = [p for p in clips["real"] if p not in {c[0] for c in chosen}]
    rows, manifest = [], []
    for number, (path, category) in enumerate(tqdm(chosen, desc="conditions")):
        feats = ClipFeatures(path, embedder, tracker)
        if feats.segments == 0:
            continue
        other = ClipFeatures(rng.choice(other_pool), embedder, tracker)
        samples = int(SEGMENT_SECONDS * WHISPER_SAMPLE_RATE)
        other_wave = np.resize(other.waveform, feats.segments * samples)
        conditions = {
            "original": score(scorer, feats.video, feats.audio_features(embedder, feats.waveform)),
            "audio_zeroed": score(scorer, feats.video, None),
            "audio_shifted_plus_200ms": score(scorer, feats.video, feats.audio_features(embedder, shift_waveform(feats.waveform, int(0.2 * WHISPER_SAMPLE_RATE)))),
            "audio_shifted_minus_200ms": score(scorer, feats.video, feats.audio_features(embedder, shift_waveform(feats.waveform, -int(0.2 * WHISPER_SAMPLE_RATE)))),
            "audio_of_another_video": score(scorer, feats.video, feats.audio_features(embedder, other_wave)),
            "video_zeroed": score(scorer, feats.video, feats.audio_features(embedder, feats.waveform), zero_video=True),
        }
        twin = fake_audio.get((path.parts[-2], path.stem))
        if category == "real" and twin is not None:
            twin_wave = extract_audio_waveform(twin, 0.0, feats.segments * SEGMENT_SECONDS, WHISPER_SAMPLE_RATE)
            conditions["fake_audio_over_real_video"] = score(scorer, feats.video, feats.audio_features(embedder, twin_wave))
        clip_id = f"conditions_{number:02d}"
        for name, (p, a) in conditions.items():
            rows.append({"clip": clip_id, "condition": name, "p_fake": p, "alpha": a})
        base = scores.get(path.stem, ("", ""))
        manifest.append({"id": clip_id, "category": category, "path": str(path), "p_fake_video_level": base[0], "alpha_video_level": base[1],
                         "fake_audio_twin": str(twin) if twin else "", "why_chosen": "seeded random draw from the test split; reals that have a fake-audio twin of the same source"})
    for name, content in (("conditions.csv", rows), ("manifest.csv", manifest)):
        with open(out_cond / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(content[0]))
            writer.writeheader()
            writer.writerows(content)
    print(f"sync: {len({r['id'] for r in lag_rows})} clips; conditions: {len(manifest)} clips")


if __name__ == "__main__":
    main()
