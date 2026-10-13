from __future__ import annotations

from typing import Callable

import numpy as np


def shift_waveform(waveform: np.ndarray, shift_samples: int) -> np.ndarray:
    """Delays the audio by shift_samples (negative advances it), keeping the length
    and filling the gap with silence."""
    shifted = np.zeros_like(waveform)
    if shift_samples >= 0:
        shifted[shift_samples:] = waveform[: len(waveform) - shift_samples]
    else:
        shifted[: len(waveform) + shift_samples] = waveform[-shift_samples:]
    return shifted


def shift_sweep(
    score_fn: Callable[[np.ndarray], float], waveform: np.ndarray, sample_rate: int, shifts_ms: list[int]
) -> list[tuple[int, float]]:
    """Scores the clip with its audio moved against the video by each offset. A
    verdict that does not move with the offset is not using audio-video alignment;
    a score that changes sharply away from zero offset is."""
    return [(ms, float(score_fn(shift_waveform(waveform, int(round(ms * sample_rate / 1000)))))) for ms in shifts_ms]
