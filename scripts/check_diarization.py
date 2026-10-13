from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.segment_embedding import WHISPER_SAMPLE_RATE
from deepfake_detection.inference.diarization import EcapaDiarizer


def load_waveforms(root: Path, limit: int, seconds: float) -> list[np.ndarray]:
    with open(root / "labels" / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:limit]
    return [extract_audio_waveform(root / row["raw_path"], 0.0, seconds, WHISPER_SAMPLE_RATE) for row in tqdm(rows, desc=root.name)]


def tally(values: list[int]) -> dict[int, int]:
    return {int(k): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the diarizer on turn-taking videos (two speakers, known turn length) and on single-voice videos.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--conversation", default="fakeavceleb_conversation")
    parser.add_argument("--single-voice", default="fakeavceleb_tiled")
    parser.add_argument("--turn-seconds", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    root = load_config(args.config).paths.datasets_root
    diarizer = EcapaDiarizer()
    sample_rate = WHISPER_SAMPLE_RATE

    conversation = load_waveforms(root / args.conversation, args.limit, 2 * args.turn_seconds)
    counts, boundaries, same, different = [], [], [], []
    for wave in conversation:
        turns = diarizer(wave, sample_rate, num_speakers=2)
        names = sorted({t.speaker for t in turns})
        counts.append(len(names))
        if len(names) == 2:
            first = turns[0].speaker
            boundaries.append(max(t.end for t in turns if t.speaker == first))
        starts, kept, embeddings = diarizer.window_embeddings(wave, sample_rate)
        unit = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity = unit @ unit.T
        centers = np.array([starts[i] + diarizer.window_seconds / 2 for i in kept])
        side = (centers >= args.turn_seconds).astype(int)
        clean = np.abs(centers - args.turn_seconds) > diarizer.window_seconds / 2
        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                if clean[i] and clean[j]:
                    (same if side[i] == side[j] else different).append(similarity[i, j])

    single = [len({t.speaker for t in diarizer(wave, sample_rate, num_speakers=2)}) for wave in load_waveforms(root / args.single_voice, args.limit, 2 * args.turn_seconds)]
    print(f"two-speaker videos ({len(counts)}): speakers found {tally(counts)}")
    if boundaries:
        print(f"first turn ends at a median of {np.median(boundaries):.2f} s (true {args.turn_seconds} s)")
    print(f"cosine similarity of windows, same speaker: mean {np.mean(same):.3f}, 5th percentile {np.percentile(same, 5):.3f}")
    print(f"cosine similarity of windows, different speakers: mean {np.mean(different):.3f}, 95th percentile {np.percentile(different, 95):.3f}")
    print(f"single-voice videos ({len(single)}): speakers found {tally(single)}")


if __name__ == "__main__":
    main()
