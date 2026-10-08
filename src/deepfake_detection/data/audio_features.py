from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
from transformers import WhisperFeatureExtractor

AV_TIME_BASE = 1_000_000


def extract_audio_waveform(path: Path, start_sec: float, end_sec: float, sample_rate: int) -> np.ndarray:
    container = av.open(str(path))
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sample_rate)

    container.seek(int(start_sec * AV_TIME_BASE))
    samples = []
    for frame in container.decode(stream):
        if frame.time is not None and frame.time >= end_sec:
            break
        for resampled in resampler.resample(frame):
            array = resampled.to_ndarray()
            if array.size:
                samples.append(array)
    container.close()

    if not samples:
        raise ValueError(f"no audio samples decoded for {path} [{start_sec}, {end_sec})")

    waveform = np.concatenate(samples, axis=1).flatten().astype(np.float32) / 32768.0
    expected_len = int(round((end_sec - start_sec) * sample_rate))
    if len(waveform) < expected_len:
        waveform = np.pad(waveform, (0, expected_len - len(waveform)))
    else:
        waveform = waveform[:expected_len]
    return waveform


def waveform_to_mel(
    waveform: np.ndarray, sample_rate: int, feature_extractor: WhisperFeatureExtractor
) -> torch.Tensor:
    """Zero-pads/truncates to Whisper's fixed 30s (3,000-mel-frame) input window."""
    features = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    return features["input_features"][0]
