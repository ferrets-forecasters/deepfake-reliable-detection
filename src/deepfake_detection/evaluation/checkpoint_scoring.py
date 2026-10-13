from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from deepfake_detection.reference import ReferenceRepo

VIDEO_DIM = 768
AUDIO_DIM = 1024


class CheckpointScorer:
    """Scores one video, given as the sequence of its segment vectors, with an exported
    checkpoint. Returns P(fake) and, for the gated model, the gate alpha. The same model
    can also be called with gradients on (fake_margin), for the explanation code."""

    def __init__(self, reference: ReferenceRepo, model_dir: Path):
        meta = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))
        self.device = reference.device
        self.model = reference.build_model(meta["model"]).to(self.device)
        state = torch.load(model_dir / "best_f1.pt", map_location=self.device, weights_only=True)
        head = getattr(self.model, "domain_head", None)
        key = "domain_head.3.weight"
        if key in state and isinstance(head, torch.nn.Sequential) and head[-1].out_features != state[key].shape[0]:
            head[-1] = torch.nn.Linear(state[key].shape[1], state[key].shape[0])
        self.model.to(self.device).load_state_dict(state)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.stats = torch.load(model_dir / "norm_stats.pt", map_location="cpu", weights_only=False)["train_stats"]
        self.has_gate = hasattr(self.model, "gate")
        self._stat_tensors = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=self.device) for k, v in self.stats.items()}

    def normalize(self, video_feats: np.ndarray, audio_feats: np.ndarray) -> np.ndarray:
        """video_feats [T, 768], audio_feats [T, 1024] to the z-scored [T, 1792] input."""
        video = (video_feats - self.stats["video_mean"]) / self.stats["video_std"]
        audio = (audio_feats - self.stats["audio_mean"]) / self.stats["audio_std"]
        return np.concatenate([video, audio], axis=1).astype(np.float32)

    def normalize_tensor(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        """Same as normalize, on tensors, so gradients reach the raw encoder outputs."""
        s = self._stat_tensors
        video = (video_feats - s["video_mean"]) / s["video_std"]
        audio = (audio_feats - s["audio_mean"]) / s["audio_std"]
        return torch.cat([video, audio], dim=-1)

    def _forward(self, x: torch.Tensor, audio_available: bool = True):
        x = x.unsqueeze(0).to(self.device)
        lengths = torch.tensor([x.shape[1]])
        mask = torch.arange(x.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1).to(self.device)
        audio_avail = torch.tensor([audio_available], device=self.device)
        if self.has_gate:
            logits, _, alpha = self.model(x, mask, lengths, lambda_=0.0, return_gate=True, audio_avail=audio_avail)
            return logits, alpha
        logits, _ = self.model(x, mask, lengths, lambda_=0.0, audio_avail=audio_avail)
        return logits, None

    @torch.no_grad()
    def score(self, x: np.ndarray, audio_available: bool = True) -> tuple[float, float | None]:
        logits, alpha = self._forward(torch.from_numpy(x), audio_available=audio_available)
        return float(torch.softmax(logits, dim=-1)[0, 1]), (float(alpha[0]) if alpha is not None else None)

    @torch.no_grad()
    def score_visual_only(self, video_feats: np.ndarray) -> tuple[float, float | None]:
        """Scores a sequence of video vectors [T, 768] with no audio, as the training code treats
        datasets without an audio signal (CelebDFv2, FF++): the audio half of the input is zero
        after normalisation and the gate weight is fixed to the visual expert."""
        video = (video_feats - self.stats["video_mean"]) / self.stats["video_std"]
        x = np.concatenate([video, np.zeros((len(video), AUDIO_DIM))], axis=1).astype(np.float32)
        logits, alpha = self._forward(torch.from_numpy(x), audio_available=False)
        return float(torch.softmax(logits, dim=-1)[0, 1]), (float(alpha[0]) if alpha is not None else None)

    def fake_margin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Differentiable fake logit minus real logit for x [T, 1792], and the gate alpha.
        The model stays in eval mode (dropout off) and cuDNN is switched off for the forward
        pass, because cuDNN's recurrent backward is only allowed in training mode."""
        with torch.backends.cudnn.flags(enabled=False):
            logits, alpha = self._forward(x)
        return logits[0, 1] - logits[0, 0], (alpha[0] if alpha is not None else None)

    def margin_from_features(self, video_feat: np.ndarray, audio_feat: np.ndarray) -> float:
        """Fake-minus-real margin of a one-segment video, without gradients."""
        with torch.no_grad():
            x = self.normalize_tensor(torch.as_tensor(video_feat[None], device=self.device), torch.as_tensor(audio_feat[None], device=self.device))
            return float(self.fake_margin(x)[0])
