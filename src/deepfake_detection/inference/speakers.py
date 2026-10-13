from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpeechTurn:
    start: float
    end: float
    speaker: str


class Diarizer(Protocol):
    """Anything that turns a waveform into speaker turns. The diarization model is chosen separately
    (open code, weights and license), this module only needs the turns."""

    def __call__(self, waveform: np.ndarray, sample_rate: int, num_speakers: int | None = None) -> list[SpeechTurn]: ...


class SingleSpeaker:
    """Default: the whole clip is one speaker, so every face gets the full audio."""

    def __call__(self, waveform: np.ndarray, sample_rate: int, num_speakers: int | None = None) -> list[SpeechTurn]:
        return [SpeechTurn(0.0, len(waveform) / sample_rate, "speaker_0")]


def speech_activity(turns: list[SpeechTurn], speaker: str, times: np.ndarray) -> np.ndarray:
    """1.0 where the speaker is talking at each time, else 0.0."""
    active = np.zeros(len(times), dtype=float)
    for turn in turns:
        if turn.speaker == speaker:
            active[(times >= turn.start) & (times < turn.end)] = 1.0
    return active


def speaker_waveform(waveform: np.ndarray, turns: list[SpeechTurn], speaker: str, sample_rate: int) -> np.ndarray:
    """The clip's audio with everything outside the speaker's turns set to silence."""
    kept = np.zeros_like(waveform)
    for turn in turns:
        if turn.speaker == speaker:
            lo, hi = int(turn.start * sample_rate), int(turn.end * sample_rate)
            kept[lo:hi] = waveform[lo:hi]
    return kept


def _zscore(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread > 1e-9 else np.zeros_like(values)


def link_speakers_to_tracks(
    turns: list[SpeechTurn],
    mouth_open_by_track: dict[int, tuple[np.ndarray, np.ndarray]],
    min_correlation: float = 0.1,
) -> dict[str, int]:
    """Matches each speaker to the face whose lip motion follows that speaker's speech.
    mouth_open_by_track maps a track id to (times in seconds, mouth opening at those times).
    Lip motion is the absolute change of the opening between samples. The match is
    one-to-one and greedy by correlation; a speaker whose best correlation is below
    min_correlation stays unlinked, and then the caller falls back to the shared audio."""
    speakers = sorted({t.speaker for t in turns})
    scores: list[tuple[float, str, int]] = []
    for track_id, (times, opening) in mouth_open_by_track.items():
        if len(times) < 3:
            continue
        motion = _zscore(np.abs(np.diff(opening)))
        midpoints = (times[:-1] + times[1:]) / 2
        for speaker in speakers:
            activity = _zscore(speech_activity(turns, speaker, midpoints))
            if not activity.any() or not motion.any():
                continue
            scores.append((float(np.mean(motion * activity)), speaker, track_id))

    linked: dict[str, int] = {}
    used_tracks: set[int] = set()
    for score, speaker, track_id in sorted(scores, reverse=True):
        if score < min_correlation:
            break
        if speaker in linked or track_id in used_tracks:
            continue
        linked[speaker] = track_id
        used_tracks.add(track_id)
    return linked
