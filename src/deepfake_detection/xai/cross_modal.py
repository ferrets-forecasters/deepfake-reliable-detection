from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from deepfake_detection.xai.cam import fold_video_tokens, token_cam
from deepfake_detection.xai.lipsync import shift_sweep

AUDIO_STEP_SECONDS = 0.02   # one Whisper encoder step covers 20 ms
VIDEO_DIM = 768


@dataclass
class SegmentExplanation:
    """Why one 2-second segment of one face got its verdict. Positive values push toward fake."""
    margin: float                  # fake logit minus real logit
    p_fake: float
    alpha: float | None            # gate weight, 0 audio expert, 1 visual expert
    video_contribution: float      # gradient x input summed over the 768 video dimensions
    audio_contribution: float      # same over the 1,024 audio dimensions
    video_share: float             # share of the absolute gradient x input mass on the video half
    audio_share: float
    video_cam: np.ndarray          # [time steps, grid, grid], exact CAM of the mean-pooled VideoMAE tokens
    audio_step_cam: np.ndarray     # [1500] exact CAM over the Whisper encoder steps (20 ms each)
    mel_saliency: np.ndarray       # [80, 3000] absolute gradient of the margin with respect to the log-mel


def explain_segment(embedder, scorer, face_frames: list[np.ndarray], waveform: np.ndarray) -> SegmentExplanation:
    """Gradients from the verdict back through the frozen encoders.

    The classifier sees only the two pooled vectors, so the video map is the exact CAM of the
    mean-pooled VideoMAE tokens and the audio map the exact CAM of the mean-pooled Whisper steps;
    the log-mel saliency goes one step further back, through the Whisper encoder. The pooled audio
    keeps the zero-padded steps of the 30 s window, as the checkpoints were trained, so the audio
    CAM shows how much the padding takes part. Backpropagating through Whisper-medium needs a few GB of GPU memory."""
    device = embedder.device
    resized = [cv2.resize(frame, (224, 224)) for frame in face_frames]
    inputs = embedder.processor(resized, return_tensors="pt").to(device)
    with torch.no_grad():
        tokens = embedder.video_model(**inputs).last_hidden_state[0]
    video_pooled = tokens.mean(dim=0).detach().requires_grad_(True)

    mel = embedder.log_mel(waveform).detach().clone().requires_grad_(True)
    audio_steps = embedder.audio_model.encoder(mel.unsqueeze(0))[0]
    audio_pooled = audio_steps.mean(dim=0)

    x = scorer.normalize_tensor(video_pooled[None], audio_pooled[None])
    margin, alpha = scorer.fake_margin(x)
    g_x, g_video, g_audio, g_mel = torch.autograd.grad(margin, [x, video_pooled, audio_pooled, mel])

    config = embedder.video_model.config
    grid = 224 // config.patch_size
    steps = tokens.shape[0] // (grid * grid)
    contribution = (g_x * x).detach()[0]
    mass = contribution.abs()
    return SegmentExplanation(
        margin=float(margin), p_fake=float(torch.sigmoid(margin)), alpha=float(alpha) if alpha is not None else None,
        video_contribution=float(contribution[:VIDEO_DIM].sum()), audio_contribution=float(contribution[VIDEO_DIM:].sum()),
        video_share=float(mass[:VIDEO_DIM].sum() / mass.sum()), audio_share=float(mass[VIDEO_DIM:].sum() / mass.sum()),
        video_cam=fold_video_tokens(token_cam(tokens, g_video), steps, grid).cpu().numpy(),
        audio_step_cam=token_cam(audio_steps.detach(), g_audio).cpu().numpy(),
        mel_saliency=g_mel.abs().cpu().numpy(),
    )


def mouth_cam_share(video_cam: np.ndarray, mouth_boxes: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """Share of the positive video CAM that falls on the mouth, and the share a uniform map would put there.
    mouth_boxes: one (x0, y0, x1, y1) per time step, as fractions of the face crop. A measured share well
    above the uniform share means the evidence sits on the lips."""
    positive = np.maximum(video_cam, 0.0)
    grid = positive.shape[-1]
    inside, area = 0.0, 0.0
    for step, (x0, y0, x1, y1) in enumerate(mouth_boxes):
        mask = np.zeros((grid, grid), dtype=bool)
        mask[int(np.floor(y0 * grid)):int(np.ceil(y1 * grid)), int(np.floor(x0 * grid)):int(np.ceil(x1 * grid))] = True
        inside += positive[step][mask].sum()
        area += mask.mean()
    total = positive[: len(mouth_boxes)].sum()
    return (float(inside / total) if total > 0 else float("nan")), float(area / len(mouth_boxes))


@dataclass
class LipAudioAlignment:
    correlation: float            # Pearson correlation between lip motion and audio relevance over the windows
    lip_motion: np.ndarray        # per window
    audio_relevance: np.ndarray   # per window


def lip_audio_alignment(
    mouth_open: np.ndarray, fps: float, audio_step_cam: np.ndarray, segment_seconds: float = 2.0, window_seconds: float = 0.25
) -> LipAudioAlignment:
    """Does the audio evidence peak when the lips move? Lip motion is the absolute frame-to-frame change of the
    mouth opening; audio relevance is the positive part of the audio step CAM. Both are summed in windows of
    window_seconds over the segment and correlated. A high value says the verdict leans on the part of the audio that
    coincides with mouth movement; it is a description of the model, not proof of synchrony."""
    windows = int(round(segment_seconds / window_seconds))
    motion_times = (np.arange(len(mouth_open) - 1) + 0.5) / fps
    motion = np.abs(np.diff(mouth_open))
    edges = np.linspace(0.0, segment_seconds, windows + 1)
    lip = np.array([motion[(motion_times >= lo) & (motion_times < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])

    steps = int(round(segment_seconds / AUDIO_STEP_SECONDS))
    positive = np.maximum(audio_step_cam[:steps], 0.0)
    step_times = (np.arange(steps) + 0.5) * AUDIO_STEP_SECONDS
    relevance = np.array([positive[(step_times >= lo) & (step_times < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])

    if lip.std() < 1e-12 or relevance.std() < 1e-12:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(lip, relevance)[0, 1])
    return LipAudioAlignment(correlation, lip, relevance)


def audio_shift_probe(embedder, scorer, video_feature: np.ndarray, waveform: np.ndarray, sample_rate: int, shifts_ms: list[int]) -> list[tuple[int, float]]:
    """Fake-minus-real margin of the segment with its audio moved against the video by each offset."""
    return shift_sweep(
        lambda shifted: scorer.margin_from_features(video_feature, embedder.audio_feature(shifted)), waveform, sample_rate, shifts_ms
    )


def mismatch_probe(embedder, scorer, video_feature: np.ndarray, other_waveform: np.ndarray) -> float:
    """Margin with the audio of a different clip. The same verdict as with the true audio means the audio is not being checked against the face."""
    return scorer.margin_from_features(video_feature, embedder.audio_feature(other_waveform))


def video_cam_for_segment(embedder, scorer, face_frames: list[np.ndarray], audio_feature: np.ndarray) -> np.ndarray:
    """The video CAM of one segment ([time steps, grid, grid], positive part) without the audio
    gradients: the margin is differentiated with respect to the pooled VideoMAE vector only, so
    Whisper is not backpropagated and the cost is one forward pass of VideoMAE plus the head."""
    resized = [cv2.resize(frame, (224, 224)) for frame in face_frames]
    inputs = embedder.processor(resized, return_tensors="pt").to(embedder.device)
    with torch.no_grad():
        tokens = embedder.video_model(**inputs).last_hidden_state[0].float()
    video_pooled = tokens.mean(dim=0).detach().requires_grad_(True)
    audio = torch.as_tensor(audio_feature, dtype=torch.float32, device=embedder.device)
    margin, _ = scorer.fake_margin(scorer.normalize_tensor(video_pooled[None], audio[None]))
    (g_video,) = torch.autograd.grad(margin, [video_pooled])
    grid = 224 // embedder.video_model.config.patch_size
    steps = tokens.shape[0] // (grid * grid)
    return fold_video_tokens(token_cam(tokens, g_video), steps, grid).clamp(min=0).cpu().numpy()
