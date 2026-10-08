from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from deepfake_detection.config import Config


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


class DomainClassifier(nn.Module):
    """DANN-style domain-adversarial branch (Ganin & Lempitsky).

    Applied to the shared 1,792-dim z-scored encoder representation whenever
    more than one source dataset is present in a training run, predicting
    which dataset a segment came from through a gradient-reversal layer so
    the shared representation is pushed toward being dataset-invariant.
    """

    def __init__(self, config: Config, num_domains: int, lambd: float = 1.0):
        super().__init__()
        self.lambd = lambd
        hidden = config.model.gru_hidden
        self.classifier = nn.Sequential(
            nn.Linear(config.model.concat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        reversed_repr = _GradientReversal.apply(shared_repr, self.lambd)
        return self.classifier(reversed_repr)
