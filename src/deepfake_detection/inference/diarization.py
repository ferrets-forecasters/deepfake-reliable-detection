from __future__ import annotations

from pathlib import Path

import numpy as np

from deepfake_detection.config.config import _project_root
from deepfake_detection.inference.speakers import SpeechTurn

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"  # Apache-2.0, public, no access token
ECAPA_DIR = _project_root() / "external" / "speechbrain" / "spkrec-ecapa-voxceleb"


def voiced_mask(waveform: np.ndarray, sample_rate: int, frame_seconds: float = 0.025, rel_threshold: float = 0.1) -> np.ndarray:
    """One boolean per frame: the frame's RMS is above rel_threshold times the loud end of the clip (95th percentile)."""
    frame = max(int(frame_seconds * sample_rate), 1)
    count = len(waveform) // frame
    if count == 0:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((waveform[: count * frame].reshape(count, frame) ** 2).mean(axis=1))
    return rms > rel_threshold * np.percentile(rms, 95)


def window_starts(total_seconds: float, window_seconds: float, hop_seconds: float) -> list[float]:
    if total_seconds <= window_seconds:
        return [0.0]
    starts = list(np.arange(0.0, total_seconds - window_seconds + 1e-9, hop_seconds))
    if starts[-1] + window_seconds < total_seconds - 1e-6:
        starts.append(total_seconds - window_seconds)
    return [float(s) for s in starts]


def labels_to_turns(starts: list[float], window_seconds: float, labels: list[int | None], total_seconds: float, slot_seconds: float) -> list[SpeechTurn]:
    """Each time slot takes the label of the window whose centre is nearest (None = silence); equal neighbours merge into one turn."""
    centers = np.array([s + window_seconds / 2 for s in starts])
    slots = int(np.ceil(total_seconds / slot_seconds))
    turns: list[SpeechTurn] = []
    for slot in range(slots):
        lo, hi = slot * slot_seconds, min((slot + 1) * slot_seconds, total_seconds)
        label = labels[int(np.argmin(np.abs(centers - (lo + hi) / 2)))]
        if label is None:
            continue
        name = f"speaker_{label}"
        if turns and turns[-1].speaker == name and abs(turns[-1].end - lo) < 1e-6:
            turns[-1] = SpeechTurn(turns[-1].start, hi, name)
        else:
            turns.append(SpeechTurn(lo, hi, name))
    return turns


class EcapaDiarizer:
    """Speaker diarization from open parts: an energy voice-activity test, ECAPA-TDNN speaker embeddings
    (SpeechBrain), and average-linkage clustering of the windows. Windows join one speaker while their
    mean cosine similarity stays above 1 - same_speaker_distance, and the number of speakers is capped
    at the number given by the caller (the number of detected faces). On 20 two-speaker FakeAVCeleb
    conversations (scripts/check_diarization.py), windows of one speaker had mean similarity 0.46 (5th percentile 0.21) and windows of
    different speakers 0.03 (95th percentile 0.18). Clips are short, so windows are 1 s with a 0.5 s hop."""

    def __init__(self, window_seconds: float = 1.0, hop_seconds: float = 0.5, min_voiced: float = 0.4, same_speaker_distance: float = 0.8, device: str | None = None, model_dir: Path = ECAPA_DIR):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        self.window_seconds, self.hop_seconds, self.min_voiced = window_seconds, hop_seconds, min_voiced
        self.same_speaker_distance = same_speaker_distance
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(source=ECAPA_SOURCE, savedir=str(model_dir), run_opts={"device": device})
        self.device = device

    def window_embeddings(self, waveform: np.ndarray, sample_rate: int) -> tuple[list[float], list[int], np.ndarray]:
        """Window start times, the indices of the windows that are voiced enough, and their ECAPA embeddings [kept, 192]."""
        starts = window_starts(len(waveform) / sample_rate, self.window_seconds, self.hop_seconds)
        voiced = voiced_mask(waveform, sample_rate)
        frames_per_second = 1.0 / 0.025
        embeddings, kept = [], []
        for index, start in enumerate(starts):
            lo, hi = int(start * sample_rate), int((start + self.window_seconds) * sample_rate)
            chunk = waveform[lo:hi]
            share = voiced[int(start * frames_per_second): int((start + self.window_seconds) * frames_per_second)].mean() if len(voiced) else 0.0
            if len(chunk) < sample_rate * 0.3 or share < self.min_voiced:
                continue
            with self._torch.no_grad():
                embedding = self.model.encode_batch(self._torch.from_numpy(chunk.astype(np.float32))[None].to(self.device))
            embeddings.append(embedding.squeeze().cpu().numpy())
            kept.append(index)
        return starts, kept, np.stack(embeddings) if embeddings else np.zeros((0, 0))

    def __call__(self, waveform: np.ndarray, sample_rate: int, num_speakers: int | None = None) -> list[SpeechTurn]:
        total = len(waveform) / sample_rate
        starts, kept, embeddings = self.window_embeddings(waveform, sample_rate)
        if not kept:
            return []

        assigned = self._cluster(embeddings, num_speakers)
        order = {}
        for label in assigned:
            order.setdefault(int(label), len(order))
        labels: list[int | None] = [None] * len(starts)
        for index, label in zip(kept, assigned):
            labels[index] = order[int(label)]
        return labels_to_turns(starts, self.window_seconds, labels, total, self.hop_seconds)

    def _cluster(self, embeddings: np.ndarray, max_speakers: int | None) -> np.ndarray:
        if len(embeddings) < 2 or max_speakers == 1:
            return np.zeros(len(embeddings), dtype=int)
        from sklearn.cluster import AgglomerativeClustering

        assigned = AgglomerativeClustering(n_clusters=None, distance_threshold=self.same_speaker_distance, metric="cosine", linkage="average").fit_predict(embeddings)
        if max_speakers and len(set(assigned.tolist())) > max_speakers:
            assigned = AgglomerativeClustering(n_clusters=max_speakers, metric="cosine", linkage="average").fit_predict(embeddings)
        return assigned
