"""
Modality Reliability Gate — the core of the MoE fusion mechanism.

The gate is a small MLP that takes two scalar anomaly scores as input and
outputs a soft weighting α ∈ [0, 1]:

    α → 0  :  audio-visual sync expert dominates
               (appropriate for TTS/voice-swap fakes like PolyglotFake,
                where the synthetic audio diverges from the natural face)

    α → 1  :  visual artifact expert dominates
               (appropriate for face-swap fakes like HiDF,
                where the audio is genuine and carries no manipulation signal)

Grounded in MoE gating theory: rather than hard-routing to one expert, the
gate produces a differentiable soft weight so gradients flow through both
experts during training, and the gate itself is trained end-to-end.

DANN training on the fused representation prevents the gate from collapsing to
a simple dataset-identity detector (PolyglotFake → 0, HiDF → 1) and forces it
to learn manipulation-type identity instead.
"""
import torch
import torch.nn as nn

from deepfake_detection.reference.config import GATE_HIDDEN


class ModalityReliabilityGate(nn.Module):
    """
    MLP gate: (audio_anomaly, visual_anomaly) → α ∈ [0, 1].

    Parameters
    ----------
    hidden_dim : int
        Hidden layer width.  Kept small intentionally — the gate input is only
        2-dimensional; a large MLP would overfit to dataset identity.
    """

    def __init__(self, hidden_dim: int = GATE_HIDDEN):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, audio_anomaly: torch.Tensor, visual_anomaly: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_anomaly  : [B]  mean cross-attention residual norm from AudioSyncExpert
        visual_anomaly : [B]  max temporal variance from VisualArtifactExpert

        Returns
        -------
        alpha : [B]  soft weight; higher means more reliance on visual expert
        """
        x     = torch.stack([audio_anomaly, visual_anomaly], dim=-1)  # [B, 2]
        alpha = self.gate(x).squeeze(-1)                               # [B]
        return alpha
