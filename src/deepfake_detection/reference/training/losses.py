"""
Loss functions for the MoE deepfake classifier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for binary classification with class-imbalance handling.

    FL(p_t) = α_t · (1 − p_t)^γ · CE(p_t)

    Parameters
    ----------
    gamma           : focusing parameter; higher = more focus on hard examples
    alpha           : weight on the *positive* (fake) class.  Set > 0.5 to
                      up-weight the minority positive class (e.g. PolyglotFake
                      has a 17:1 fake/real imbalance).
    label_smoothing : applied before computing CE to prevent over-confident logits
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce      = F.cross_entropy(logits, targets,
                                  label_smoothing=self.label_smoothing,
                                  reduction="none")
        pt      = torch.exp(-ce)
        alpha_t = torch.where(
            targets == 1,
            torch.tensor(self.alpha,       device=logits.device),
            torch.tensor(1.0 - self.alpha, device=logits.device),
        )
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()
