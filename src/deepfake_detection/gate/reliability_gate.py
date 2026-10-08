from __future__ import annotations

import torch
from torch import nn

from deepfake_detection.config import Config


class ModalityReliabilityGate(nn.Module):
    """Three-layer MLP gate: [audio_anomaly, visual_anomaly] -> alpha in [0, 1].

    alpha -> 1 favours the visual expert, alpha -> 0 favours the audio expert.
    """

    def __init__(self, config: Config):
        super().__init__()
        hidden = config.model.gate_hidden
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio_anomaly: torch.Tensor, visual_anomaly: torch.Tensor) -> torch.Tensor:
        s = torch.stack([audio_anomaly, visual_anomaly], dim=-1)
        alpha = self.net(s).squeeze(-1)
        return alpha
