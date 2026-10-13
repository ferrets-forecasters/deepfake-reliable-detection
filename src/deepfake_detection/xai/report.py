from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deepfake_detection.xai.cross_modal import AUDIO_STEP_SECONDS, LipAudioAlignment, SegmentExplanation
from deepfake_detection.xai.overlay import overlay_video_cam

INK, MARK = "#222222", "#c0392b"


def save_video_cam(face_frames: list[np.ndarray], cam: np.ndarray, path: Path, columns: int = 4) -> None:
    """One tile per CAM time step (the first of its two frames), all on one colour scale."""
    resized = [cv2.resize(frame, (224, 224)) for frame in face_frames]
    shown = overlay_video_cam(resized, np.maximum(cam, 0.0))[::2]
    rows = [np.concatenate(shown[i:i + columns], axis=1) for i in range(0, len(shown), columns)]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def save_audio_saliency(explanation: SegmentExplanation, path: Path, segment_seconds: float = 2.0) -> None:
    frames = int(segment_seconds * 100)
    steps = int(segment_seconds / AUDIO_STEP_SECONDS)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    top.imshow(explanation.mel_saliency[:, :frames], aspect="auto", origin="lower", extent=[0, segment_seconds, 0, 80], cmap="magma")
    top.set_ylabel("mel bin")
    bottom.plot(np.arange(steps) * AUDIO_STEP_SECONDS, explanation.audio_step_cam[:steps], color=INK, linewidth=1)
    bottom.axhline(0, color="#999999", linewidth=0.5)
    bottom.set_xlabel("seconds")
    bottom.set_ylabel("CAM")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_branches(explanation: SegmentExplanation, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(4, 2.6))
    axis.barh(["video", "audio"], [explanation.video_contribution, explanation.audio_contribution], color=[INK, MARK])
    axis.axvline(0, color="#999999", linewidth=0.5)
    axis.set_xlabel("gradient x input")
    alpha = "" if explanation.alpha is None else f", alpha {explanation.alpha:.2f}"
    axis.set_title(f"P(fake) {explanation.p_fake:.2f}{alpha}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_shift_sweep(rows: list[tuple[int, float]], path: Path, mismatch_margin: float | None = None) -> None:
    figure, axis = plt.subplots(figsize=(5, 3))
    ordered = sorted(rows)
    axis.plot([ms for ms, _ in ordered], [m for _, m in ordered], marker="o", color=INK)
    if mismatch_margin is not None:
        axis.axhline(mismatch_margin, color=MARK, linestyle="--", linewidth=1, label="other clip's audio")
        axis.legend(frameon=False)
    axis.set_xlabel("audio shift (ms)")
    axis.set_ylabel("fake - real margin")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_explanation(
    explanation: SegmentExplanation, face_frames: list[np.ndarray], out_dir: Path, shifts: list[tuple[int, float]] | None = None,
    mismatch_margin: float | None = None, alignment: LipAudioAlignment | None = None, mouth_share: tuple[float, float] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_video_cam(face_frames, explanation.video_cam, out_dir / "video_cam.png")
    save_audio_saliency(explanation, out_dir / "audio_saliency.png")
    save_branches(explanation, out_dir / "branches.png")
    if shifts:
        save_shift_sweep(shifts, out_dir / "shift_sweep.png", mismatch_margin)
    summary = {
        key: value for key, value in asdict(explanation).items() if not isinstance(value, np.ndarray)
    }
    summary["shift_sweep"] = shifts
    summary["mismatch_margin"] = mismatch_margin
    summary["lip_audio_correlation"] = None if alignment is None else alignment.correlation
    summary["mouth_cam_share"] = None if mouth_share is None else {"measured": mouth_share[0], "uniform": mouth_share[1]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
