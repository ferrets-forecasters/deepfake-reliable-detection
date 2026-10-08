from __future__ import annotations

import torch
from torch import nn


class FocalLoss(nn.Module):
    """Binary focal loss on logits, with label smoothing applied before the focal weighting."""

    def __init__(self, gamma: float, alpha: float, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        p = torch.sigmoid(logits)
        p_t = p * smoothed + (1 - p) * (1 - smoothed)
        alpha_t = self.alpha * smoothed + (1 - self.alpha) * (1 - smoothed)
        loss = -alpha_t * (1 - p_t).pow(self.gamma) * torch.log(p_t.clamp_min(1e-8))
        return loss.mean()


class GateRoutingAuxLoss(nn.Module):
    """Per-expert supervision (weight applied by the caller): each expert's own
    logit is pushed toward the true label directly, independent of the gate's
    routing weight, so both experts keep receiving a training signal even when
    the gate routes most weight to the other one."""

    def __init__(self, gamma: float, alpha: float, label_smoothing: float = 0.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, label_smoothing=label_smoothing)

    def forward(self, visual_logit: torch.Tensor, audio_logit: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.focal(visual_logit, targets) + self.focal(audio_logit, targets))


class DomainAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, domain_logits: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(domain_logits, domain_labels)
