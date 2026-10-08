from __future__ import annotations

import torch
from torch import nn

from deepfake_detection.config import Config


class VisualArtifactExpertOutput:
    def __init__(self, logit: torch.Tensor, anomaly: torch.Tensor):
        self.logit = logit
        self.anomaly = anomaly


class VisualArtifactExpert(nn.Module):
    """Visual-branch anomaly expert.

    Operates on the temporal mean and variance of the VideoMAE per-tubelet
    embeddings alone, no audio input. The visual anomaly score the gate
    consumes is the maximum temporal variance across the 768 VideoMAE
    feature dimensions.
    """

    def __init__(self, config: Config):
        super().__init__()
        visual_dim = config.encoders.visual.hidden_size
        hidden = config.model.visual_expert_hidden
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, visual_seq: torch.Tensor) -> VisualArtifactExpertOutput:
        temporal_mean = visual_seq.mean(dim=1)
        temporal_var = visual_seq.var(dim=1, unbiased=False)
        anomaly = temporal_var.max(dim=-1).values

        features = torch.cat([temporal_mean, temporal_var], dim=-1)
        encoded = self.mlp(features)
        logit = self.classifier(encoded).squeeze(-1)

        return VisualArtifactExpertOutput(logit=logit, anomaly=anomaly)
