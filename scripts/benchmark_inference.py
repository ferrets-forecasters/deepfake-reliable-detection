from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import FaceLandmarkDetector
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, WHISPER_SAMPLE_RATE, SegmentEmbedder
from deepfake_detection.data.tiling import index_clips
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.reference import load_reference_repo

SEGMENT_SECONDS = 2.0


def median_ms(fn, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Time each stage of the pipeline per 2-second segment and the classifier at several sequence lengths, then the real-time factor of a clip of a given length.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--models", nargs="+", default=["moe_combined", "concat_combined"])
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 10, 30, 60])
    parser.add_argument("--clip-seconds", type=float, default=120.0, help="clip length used for the real-time factor")
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    index = index_clips(config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2")
    clip = next(iter(sorted(p for (kind, _, _), paths in index.items() if kind == "real" for p in paths)))

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while len(frames) < int(SEGMENT_SECONDS * fps):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    waveform = extract_audio_waveform(clip, 0.0, SEGMENT_SECONDS, WHISPER_SAMPLE_RATE)
    sampled = [frames[i] for i in np.linspace(0, len(frames) - 1, VIDEOMAE_FRAMES).round().astype(int)]

    rows = {}
    detector = FaceLandmarkDetector(max_faces=1, margin=0.25)
    detections_per_segment = int(round(SEGMENT_SECONDS * 8))
    rows["face detection, one frame (ms)"] = median_ms(lambda: detector.detect(frames[0]), args.repeats)
    rows["face detection, 8 per second, one segment (ms)"] = rows["face detection, one frame (ms)"] * detections_per_segment

    for half in (False, True):
        embedder = SegmentEmbedder(half=half)
        label = "fp16" if embedder.half else "fp32"
        rows[f"VideoMAE-base, one segment, {label} (ms)"] = median_ms(lambda: embedder.video_features([sampled]), args.repeats)
        rows[f"Whisper-medium encoder, one segment, {label} (ms)"] = median_ms(lambda: embedder.audio_features([waveform]), args.repeats)
        del embedder
        torch.cuda.empty_cache()

    scorers = {name: CheckpointScorer(reference, config.paths.checkpoints_dir(name)) for name in args.models}
    for name, scorer in scorers.items():
        for length in args.lengths:
            x = np.random.default_rng(0).standard_normal((length, 1792)).astype(np.float32)
            rows[f"classifier {name}, {length} segments (ms)"] = median_ms(lambda: scorer.score(x), args.repeats * 3)

    segments = int(args.clip_seconds // SEGMENT_SECONDS)
    per_segment_ms = rows["face detection, 8 per second, one segment (ms)"] + rows["VideoMAE-base, one segment, fp16 (ms)"] + rows["Whisper-medium encoder, one segment, fp16 (ms)"]
    for name in args.models:
        total_ms = per_segment_ms * segments + rows[f"classifier {name}, {min(args.lengths, key=lambda n: abs(n - segments))} segments (ms)"]
        rows[f"{name}: {args.clip_seconds:.0f} s clip, {segments} segments, total (ms)"] = total_ms
        rows[f"{name}: real-time factor"] = total_ms / 1000 / args.clip_seconds

    out_dir = config.paths.results_dir("benchmark_inference_speed")
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    summary = {"device": gpu, "clip": str(clip.name), "segment_seconds": SEGMENT_SECONDS, "repeats": args.repeats,
               "timings": {k: round(v, 3) for k, v in rows.items()}}
    (out_dir / "timings.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"device {gpu}")
    for key, value in summary["timings"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
