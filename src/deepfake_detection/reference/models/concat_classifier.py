"""
ConcatFusionClassifier — late-fusion ablation baseline.

This reproduces the original fixed CrossModalGRU paradigm (pre-MoE):
    1. Project video and audio to HIDDEN_DIM each
    2. Apply CrossModalAttention (audio-video cross-attend)
    3. Concatenate + cosine similarity channel
    4. Run through a pluggable sequence backbone (GRU / LSTM / Transformer)
    5. AttentionPool → classification head

Unlike MoEFusionClassifier, the fusion weight is *fixed* — audio and video
are always combined equally regardless of which modality is actually manipulated.

Role in the experimental design
────────────────────────────────
Ablation: does adaptive MoE gating actually improve over fixed late-fusion?
    concat_gru       vs   moe_gru           → effect of adaptive gate
    concat_lstm      vs   moe_lstm          → same comparison with LSTM backbone
    concat_transformer vs moe_transformer  → same with Transformer

Interface
─────────
Identical to MoEFusionClassifier:
    model(x, mask, lengths, lambda_=0.0) → (logit [B,2], domain_logit [B,2] | None)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake_detection.reference.config import AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM, DROPOUT
from deepfake_detection.reference.models.backbones import build_backbone
from deepfake_detection.reference.models.components import CrossModalAttention, AttentionPooling, GradientReversalFn


class ConcatFusionClassifier(nn.Module):
    """
    Fixed late-fusion deepfake classifier (ablation baseline).

    Parameters
    ----------
    backbone   : "gru" | "lstm" | "transformer"
    hidden_dim : projection and backbone hidden size
    num_layers : GRU/LSTM layers or Transformer depth
    dropout    : dropout probability
    """

    def __init__(self, backbone: str = "gru", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.audio_enc = nn.Sequential(
            nn.Linear(AUDIO_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.video_enc = nn.Sequential(
            nn.Linear(VIDEO_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.cross_attn = CrossModalAttention(hidden_dim)

        # backbone input: [audio | video | cosine_sim] = hidden_dim*2 + 1
        backbone_in  = hidden_dim * 2 + 1
        self.backbone = build_backbone(backbone, backbone_in, hidden_dim, num_layers, dropout)

        self.attn_pool = AttentionPooling(self.backbone.out_dim)

        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.domain_head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, mask, lengths, lambda_: float = 0.0, audio_avail=None):
        """
        Parameters
        ----------
        x       : [B, T, VIDEO_DIM + AUDIO_DIM]
        mask    : [B, T] bool
        lengths : [B] int
        lambda_ : DANN gradient reversal coefficient
        """
        video = x[..., :VIDEO_DIM]
        audio = x[..., VIDEO_DIM:]

        a = self.audio_enc(audio)
        v = self.video_enc(video)

        a_out, v_out, _, _ = self.cross_attn(a, v, mask)

        cos_sim  = F.cosine_similarity(a_out, v_out, dim=-1, eps=1e-8).unsqueeze(-1)
        fused    = torch.cat([a_out, v_out, cos_sim], dim=-1)      # [B, T, H*2+1]
        enc      = self.backbone(fused, mask, lengths)              # [B, T, out_dim]
        repr_    = self.attn_pool(enc, mask[:, :enc.shape[1]])      # [B, out_dim]
        logit    = self.head(repr_)

        if lambda_ > 0.0:
            h_rev        = GradientReversalFn.apply(repr_, lambda_)
            domain_logit = self.domain_head(h_rev)
            return logit, domain_logit

        return logit, None
