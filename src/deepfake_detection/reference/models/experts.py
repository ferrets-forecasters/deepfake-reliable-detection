"""
Two expert pathways for the Modality-Aware Adaptive Fusion classifier.

AudioSyncExpert
    Detects audio-visual synchrony violations.
    Effective on TTS-based fakes where the synthetic voice diverges from the
    natural face motion (e.g. PolyglotFake).  Uses a *pluggable* sequence
    backbone (GRU / LSTM / Transformer) for the temporal encoding step, so
    the same expert can be evaluated with different inductive biases without
    changing any other part of the architecture.

VisualArtifactExpert
    Detects spatial flickering and boundary inconsistencies in the video stream.
    Effective on face-swap fakes where the audio is untouched (e.g. HiDF).
    Temporal variance of VideoMAE embeddings serves as the primary anomaly
    signal — high variance across segments indicates unstable generated faces.
    This expert has no recurrent component; it aggregates statistics globally.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake_detection.reference.config import AUDIO_DIM, VIDEO_DIM, HIDDEN_DIM, GRU_LAYERS, DROPOUT
from deepfake_detection.reference.models.backbones import build_backbone
from deepfake_detection.reference.models.components import CrossModalAttention, AttentionPooling


class AudioSyncExpert(nn.Module):
    """
    Audio-visual synchrony pathway.

    Architecture:
        Linear projection → CrossModalAttention → Transformer pre-encoder
        → [backbone] → AttentionPool → classification head

    The Transformer pre-encoder layer was retained from the original CrossModalGRU
    as it improves the quality of the fused representation before the backbone.

    Parameters
    ----------
    backbone   : "gru" | "lstm" | "transformer"
    hidden_dim : projection hidden size (also backbone hidden size)
    num_layers : depth for GRU/LSTM layers, or Transformer layers in backbone
    dropout    : dropout probability

    Returns (per forward call)
    --------------------------
    logit         : [B, 2]    expert classification logits
    repr_         : [B, D]    pooled representation (for DANN soft-mix)
    audio_anomaly : [B]       mean norm of cross-attention residual (residual_a)
    """

    def __init__(self, backbone: str = "gru", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = GRU_LAYERS, dropout: float = DROPOUT):
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

        # Pre-encoder Transformer over the fused [audio | video] representation
        fused_dim = hidden_dim * 2
        self.pre_encoder = nn.TransformerEncoderLayer(
            d_model=fused_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )

        # backbone input: [fused | cosine_sim] = fused_dim + 1
        backbone_in   = fused_dim + 1
        self.backbone = build_backbone(backbone, backbone_in, hidden_dim, num_layers, dropout)

        self.attn_pool = AttentionPooling(self.backbone.out_dim)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, audio, video, mask, lengths):
        """
        Parameters
        ----------
        audio   : [B, T, AUDIO_DIM]
        video   : [B, T, VIDEO_DIM]
        mask    : [B, T] bool  True = valid token
        lengths : [B]    int   unpadded sequence lengths
        """
        a = self.audio_enc(audio)   # [B, T, H]
        v = self.video_enc(video)   # [B, T, H]

        # Cross-modal attention; raw outputs exposed for anomaly scoring
        a_out, v_out, a_raw_attn, _ = self.cross_attn(a, v, mask)

        # Audio anomaly score: mean temporal norm of residual
        # residual_a = a_enc - cross_attend(video→audio)
        # High values → audio is unexpectedly divergent from video predictions
        residual_a    = (a - a_raw_attn).norm(dim=-1)   # [B, T]
        audio_anomaly = residual_a.mean(dim=-1)          # [B]

        # Cosine similarity channel (explicit sync signal)
        cos_sim = F.cosine_similarity(a_out, v_out, dim=-1, eps=1e-8).unsqueeze(-1)

        fused     = torch.cat([a_out, v_out], dim=-1)                  # [B, T, H*2]
        fused     = self.pre_encoder(fused, src_key_padding_mask=~mask)
        fused_cos = torch.cat([fused, cos_sim], dim=-1)                # [B, T, H*2+1]

        enc   = self.backbone(fused_cos, mask, lengths)                # [B, T, out_dim]
        repr_ = self.attn_pool(enc, mask[:, :enc.shape[1]])            # [B, out_dim]
        logit = self.head(repr_)                                       # [B, 2]

        return logit, repr_, audio_anomaly


class VisualArtifactExpert(nn.Module):
    """
    Visual artifact pathway.

    Architecture:
        Temporal mean + temporal variance of VideoMAE embeddings
        → MLP encoder → classification head

    No sequence backbone here by design: the core signal is *global* temporal
    instability (how much do VideoMAE embeddings vary across the clip?), not
    sequential patterns.  A heavier temporal model would likely overfit to
    video-content variation rather than artifact variation.

    Returns (per forward call)
    --------------------------
    logit          : [B, 2]    expert classification logits
    repr_          : [B, H]    encoded representation (for DANN soft-mix)
    visual_anomaly : [B]       max temporal-variance value (anomaly score for gate)
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(VIDEO_DIM * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),    nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, video, mask):
        """
        Parameters
        ----------
        video : [B, T, VIDEO_DIM]
        mask  : [B, T] bool  True = valid token
        """
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).float()    # [B, 1]
        v_mask  = video * mask.unsqueeze(-1).float()                     # zero padding

        # Temporal mean (overall face appearance)
        v_mean = v_mask.sum(dim=1) / lengths                              # [B, D]

        # Temporal variance (flickering / boundary instability)
        diff  = (v_mask - v_mean.unsqueeze(1)) * mask.unsqueeze(-1).float()
        v_var = (diff ** 2).sum(dim=1) / lengths                          # [B, D]

        # Visual anomaly score: maximum variance across feature dimensions.
        # High score = at least one VideoMAE channel fluctuates strongly across
        # the clip, the hallmark of inconsistent face-swap generations (e.g. HiDF).
        visual_anomaly = v_var.max(dim=-1).values                         # [B]

        feat  = torch.cat([v_mean, v_var], dim=-1)   # [B, D*2]
        repr_ = self.encoder(feat)                   # [B, H]
        logit = self.head(repr_)                     # [B, 2]

        return logit, repr_, visual_anomaly
