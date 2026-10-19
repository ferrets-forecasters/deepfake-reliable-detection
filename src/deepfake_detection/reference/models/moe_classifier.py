"""
MoEFusionClassifier — Modality-Aware Adaptive Fusion Classifier.

Architecture overview
─────────────────────
                       ┌─────────────────────────────────────────────┐
  video [B,T,768]  ──► │  VisualArtifactExpert                      │ ──► visual_logit, visual_repr, visual_anomaly
  audio [B,T,1024] ──► │  AudioSyncExpert(backbone=gru|lstm|xfmr)   │ ──► sync_logit,   sync_repr,   audio_anomaly
                       └─────────────────────────────────────────────┘
                                           │
                              audio_anomaly, visual_anomaly
                                           │
                             ┌─────────────────────────┐
                             │  ModalityReliabilityGate │ ──► α ∈ [0,1]
                             └─────────────────────────┘
                                           │
               fused_logit = (1−α)·sync_logit  +  α·visual_logit
               fused_repr  = (1−α)·sync_repr   +  α·visual_repr_projected
                                           │
                        ┌──────────────────┴──────────────────┐
                  classification                       DANN domain head
                   (focal loss)             (gradient reversal + cross-entropy)

Manipulation-type caveat
─────────────────────────
    PolyglotFake fakes : audio + video both manipulated (TTS audio + lip-sync)
        Expected gate behaviour: α → 0  (sync expert dominates)
    HiDF fakes         : video only manipulated (face-swap), audio is genuine
        Expected gate behaviour: α → 1  (visual expert dominates)

The gate supervision auxiliary loss (in engine.py) pushes α toward these
targets during training when manipulation_type labels are available.

Forward interface
─────────────────
    model(x, mask, lengths, lambda_=0.0) → (logit [B,2], domain_logit [B,2] | None)

When return_gate=True:
    model(x, mask, lengths, lambda_=0.0, return_gate=True)
        → (logit, domain_logit, alpha [B])
    Used by engine.py to compute the gate supervision loss.
"""
import torch
import torch.nn as nn

from deepfake_detection.reference.config import HIDDEN_DIM, DROPOUT, GRU_LAYERS
from deepfake_detection.reference.models.components import GradientReversalFn
from deepfake_detection.reference.models.experts import AudioSyncExpert, VisualArtifactExpert
from deepfake_detection.reference.models.gate import ModalityReliabilityGate
from deepfake_detection.reference.config import VIDEO_DIM, AUDIO_DIM


class MoEFusionClassifier(nn.Module):
    """
    End-to-end trainable Mixture-of-Experts deepfake classifier.

    Parameters
    ----------
    backbone   : sequence encoder for AudioSyncExpert — "gru" | "lstm" | "transformer"
    hidden_dim : internal projection + backbone hidden size
    num_layers : GRU/LSTM layer count, or Transformer depth
    dropout    : dropout probability
    gate_hidden: hidden size of the ModalityReliabilityGate MLP
    """

    def __init__(self, backbone: str = "gru", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = GRU_LAYERS, dropout: float = DROPOUT,
                 gate_hidden: int = 64):
        super().__init__()
        self.sync_expert   = AudioSyncExpert(backbone, hidden_dim, num_layers, dropout)
        self.visual_expert = VisualArtifactExpert(hidden_dim, dropout)
        self.gate          = ModalityReliabilityGate(gate_hidden)

        # Project visual_repr (hidden_dim) up to sync_repr dim (backbone.out_dim)
        sync_repr_dim = self.sync_expert.backbone.out_dim
        self.visual_repr_proj = nn.Linear(hidden_dim, sync_repr_dim)

        self.domain_head = nn.Sequential(
            nn.Linear(sync_repr_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, mask, lengths, lambda_: float = 0.0,
                return_gate: bool = False, audio_avail=None):
        """
        Parameters
        ----------
        x           : [B, T, VIDEO_DIM + AUDIO_DIM]
        mask        : [B, T] bool  True = valid token
        lengths     : [B] int      unpadded sequence lengths
        lambda_     : float        DANN gradient reversal strength (0 = disabled)
        return_gate : bool         if True, also return alpha [B] as 3rd element
        audio_avail : BoolTensor [B] or None
                      When False for a sample, the gate is overridden to α=1.0
                      (route entirely through VisualArtifactExpert).
                      Used for datasets with no meaningful audio track.

        Returns
        -------
        fused_logit  : [B, 2]
        domain_logit : [B, 2] or None
        alpha        : [B]  (only when return_gate=True)
        """
        video = x[..., :VIDEO_DIM]
        audio = x[..., VIDEO_DIM:]

        # ── Expert forward passes ─────────────────────────────────────────────
        sync_logit,   sync_repr,   audio_anomaly  = self.sync_expert(audio, video, mask, lengths)
        visual_logit, visual_repr, visual_anomaly = self.visual_expert(video, mask)

        # ── Gate ──────────────────────────────────────────────────────────────
        alpha = self.gate(audio_anomaly, visual_anomaly)   # [B]

        # Override α=1.0 for audio-unavailable samples (CelebDFv2, FF++)
        if audio_avail is not None:
            unavail = ~audio_avail.to(alpha.device).bool()
            if unavail.any():
                alpha = alpha.clone()
                alpha[unavail] = 1.0

        alpha_e = alpha.unsqueeze(-1)                         # [B, 1]

        # ── Soft expert fusion ────────────────────────────────────────────────
        fused_logit = (1.0 - alpha_e) * sync_logit + alpha_e * visual_logit

        # ── DANN ──────────────────────────────────────────────────────────────
        domain_logit = None
        if lambda_ > 0.0:
            visual_repr_up = self.visual_repr_proj(visual_repr)
            fused_repr     = (1.0 - alpha_e) * sync_repr + alpha_e * visual_repr_up
            h_rev          = GradientReversalFn.apply(fused_repr, lambda_)
            domain_logit   = self.domain_head(h_rev)

        if return_gate:
            return fused_logit, domain_logit, alpha
        return fused_logit, domain_logit

    @torch.no_grad()
    def get_gate_diagnostics(self, x, mask, lengths):
        """Return per-sample gate internals for diagnostic analysis."""
        video = x[..., :VIDEO_DIM]
        audio = x[..., VIDEO_DIM:]
        _, _, audio_anomaly  = self.sync_expert(audio, video, mask, lengths)
        _, _, visual_anomaly = self.visual_expert(video, mask)
        alpha                = self.gate(audio_anomaly, visual_anomaly)
        return {
            "audio_anomaly":  audio_anomaly.cpu(),
            "visual_anomaly": visual_anomaly.cpu(),
            "alpha":          alpha.cpu(),
        }
