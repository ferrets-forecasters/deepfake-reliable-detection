from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, WhisperFeatureExtractor

from deepfake_detection.config import load_config
from deepfake_detection.data import (
    FaceBoxTracker,
    extract_audio_waveform,
    extract_video_frames,
    read_manifest,
    waveform_to_mel,
)
from deepfake_detection.encoders import AudioEncoder, VisualEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen encoders once per segment and cache the results.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default="combined.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    visual_encoder = VisualEncoder(config).to(device)
    audio_encoder = AudioEncoder(config).to(device)
    image_processor = VideoMAEImageProcessor.from_pretrained(config.encoders.visual.model_id)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.encoders.audio.model_id)
    face_tracker = FaceBoxTracker(detector_fps=config.preprocessing.video.face_detector_fps)

    manifest = read_manifest(config.paths.manifests_dir / args.manifest)
    features_dir = config.paths.features_dir

    pooled_by_segment: dict[str, np.ndarray] = {}
    skipped = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="extracting features"):
        out_dir = features_dir / row["dataset"]
        out_path = out_dir / f"{row['segment_id']}.npz"
        if out_path.exists():
            continue

        try:
            pixel_values = extract_video_frames(
                row["video_path"],
                row["start_sec"],
                row["end_sec"],
                config.preprocessing.video.num_frames,
                face_tracker,
                image_processor,
            ).unsqueeze(0).to(device)
            waveform = extract_audio_waveform(
                row["audio_path"], row["start_sec"], row["end_sec"], config.preprocessing.audio.sample_rate
            )
            input_features = waveform_to_mel(
                waveform, config.preprocessing.audio.sample_rate, feature_extractor
            ).unsqueeze(0).to(device)
        except ValueError as exc:
            tqdm.write(f"skip {row['segment_id']}: {exc}")
            skipped += 1
            continue

        visual_out = visual_encoder(pixel_values)
        audio_out = audio_encoder(input_features)

        out_dir.mkdir(parents=True, exist_ok=True)
        visual_seq = visual_out.temporal_seq[0].cpu().numpy()
        audio_seq = audio_out.temporal_seq[0].cpu().numpy()
        pooled = torch.cat([visual_out.pooled[0], audio_out.pooled[0]]).cpu().numpy()

        np.savez_compressed(out_path, visual=visual_seq, audio=audio_seq, pooled_raw=pooled)
        pooled_by_segment[row["segment_id"]] = pooled

    print(f"extracted {len(pooled_by_segment)} new segments, skipped {skipped}")
    _finalize_normalization(manifest, features_dir, config)


def _finalize_normalization(manifest, features_dir, config) -> None:
    """Computes z-score stats from the train split's pooled vectors, writes a
    normalized `concat` array into every segment's cache file."""
    train_rows = manifest[manifest["split"] == "train"]
    train_pooled = []
    for _, row in tqdm(train_rows.iterrows(), total=len(train_rows), desc="collecting train stats"):
        path = features_dir / row["dataset"] / f"{row['segment_id']}.npz"
        if path.exists():
            train_pooled.append(np.load(path)["pooled_raw"])

    stats = {
        "mean": np.mean(train_pooled, axis=0).tolist(),
        "std": (np.std(train_pooled, axis=0) + 1e-6).tolist(),
    }
    stats_path = features_dir / "norm_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="writing normalized concat"):
        path = features_dir / row["dataset"] / f"{row['segment_id']}.npz"
        if not path.exists():
            continue
        cached = dict(np.load(path))
        cached["concat"] = (cached["pooled_raw"] - mean) / std
        np.savez_compressed(path, **cached)


if __name__ == "__main__":
    main()
