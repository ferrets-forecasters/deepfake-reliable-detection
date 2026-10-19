"""
Model factory — instantiate any supported classifier from a config dict.

Called by experiments/run.py with the ``model`` section of the YAML config:

    model = build_model(cfg["model"])

Supported model types
─────────────────────
  moe              Full MoEFusionClassifier (AudioSyncExpert + VisualArtifactExpert + Gate)
                   Primary architecture.

  concat           ConcatFusionClassifier — fixed late-fusion baseline.
                   Ablation: does adaptive gating actually help?

  audio_sync_only  AudioSyncExpert alone wrapped in a standard interface.
                   Ablation: what does the sync pathway contribute on its own?

  visual_only      VisualArtifactExpert alone wrapped in a standard interface.
                   Ablation: what does the visual pathway contribute on its own?

Supported backbone values (applies to moe, concat, audio_sync_only)
───────────────────────────────────────────────────────────────────────────────
  gru              BiGRU with skip-connection residual  (default, fastest)
  lstm             BiLSTM with skip-connection residual
  transformer      Pure attention stack with positional encoding (no recurrence)
"""
import torch.nn as nn

from deepfake_detection.reference.config import HIDDEN_DIM, GRU_LAYERS, DROPOUT
from deepfake_detection.reference.models.moe_classifier import MoEFusionClassifier
from deepfake_detection.reference.models.concat_classifier import ConcatFusionClassifier
from deepfake_detection.reference.models.experts import AudioSyncExpert, VisualArtifactExpert
from deepfake_detection.reference.models.components import GradientReversalFn
from deepfake_detection.reference.config import VIDEO_DIM, AUDIO_DIM
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Ablation wrappers
# ─────────────────────────────────────────────────────────────────────────────

class AudioSyncOnlyClassifier(nn.Module):
    """
    AudioSyncExpert wrapped to match the standard model interface.

    Exposes only the audio-visual sync pathway — useful for measuring the
    contribution of that expert in isolation (no visual artifact branch, no gate).
    """

    def __init__(self, backbone: str = "gru", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = GRU_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.expert = AudioSyncExpert(backbone, hidden_dim, num_layers, dropout)
        domain_repr_dim = self.expert.backbone.out_dim
        self.domain_head = nn.Sequential(
            nn.Linear(domain_repr_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, mask, lengths, lambda_: float = 0.0, audio_avail=None):
        video  = x[..., :VIDEO_DIM]
        audio  = x[..., VIDEO_DIM:]
        logit, repr_, _ = self.expert(audio, video, mask, lengths)

        if lambda_ > 0.0:
            h_rev        = GradientReversalFn.apply(repr_, lambda_)
            domain_logit = self.domain_head(h_rev)
            return logit, domain_logit

        return logit, None


class VisualOnlyClassifier(nn.Module):
    """
    VisualArtifactExpert wrapped to match the standard model interface.

    Exposes only the visual artifact pathway — useful for measuring the
    contribution of that expert in isolation (no audio branch, no gate).
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.expert = VisualArtifactExpert(hidden_dim, dropout)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, mask, lengths, lambda_: float = 0.0, audio_avail=None):
        video  = x[..., :VIDEO_DIM]
        logit, repr_, _ = self.expert(video, mask)

        if lambda_ > 0.0:
            h_rev        = GradientReversalFn.apply(repr_, lambda_)
            domain_logit = self.domain_head(h_rev)
            return logit, domain_logit

        return logit, None


class PureAudioClassifier(nn.Module):
    """
    True unimodal audio-only baseline.
    Slices only the audio features (x[..., VIDEO_DIM:]), projects, runs GRU, classifies.
    No video features used at any point.
    """
    def __init__(self, backbone: str = "gru", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = GRU_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        from deepfake_detection.reference.models.backbones import build_backbone
        from deepfake_detection.reference.models.components import AttentionPooling
        self.enc = nn.Sequential(
            nn.Linear(AUDIO_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.backbone  = build_backbone(backbone, hidden_dim, hidden_dim, num_layers, dropout)
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
        audio = x[..., VIDEO_DIM:]
        h = self.enc(audio)
        h = self.backbone(h, mask, lengths)
        repr_ = self.attn_pool(h, mask)
        logit = self.head(repr_)
        if lambda_ > 0.0:
            h_rev = GradientReversalFn.apply(repr_, lambda_)
            return logit, self.domain_head(h_rev)
        return logit, None


class PureVideoClassifier(nn.Module):
    """
    True unimodal video-only baseline.
    Slices only the video features (x[..., :VIDEO_DIM]), projects, runs GRU, classifies.
    No audio features used at any point.
    """
    def __init__(self, backbone: str = "gru", hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = GRU_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        from deepfake_detection.reference.models.backbones import build_backbone
        from deepfake_detection.reference.models.components import AttentionPooling
        self.enc = nn.Sequential(
            nn.Linear(VIDEO_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.backbone  = build_backbone(backbone, hidden_dim, hidden_dim, num_layers, dropout)
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
        video = x[..., :VIDEO_DIM]
        h = self.enc(video)
        h = self.backbone(h, mask, lengths)
        repr_ = self.attn_pool(h, mask)
        logit = self.head(repr_)
        if lambda_ > 0.0:
            h_rev = GradientReversalFn.apply(repr_, lambda_)
            return logit, self.domain_head(h_rev)
        return logit, None


# ─────────────────────────────────────────────────────────────────────────────
# Registry & factory function
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "moe":              MoEFusionClassifier,
    "concat":           ConcatFusionClassifier,
    "audio_sync_only":  AudioSyncOnlyClassifier,
    "visual_only":      VisualOnlyClassifier,
    "pure_audio":       PureAudioClassifier,
    "pure_video":       PureVideoClassifier,
}


def build_model(model_cfg: dict) -> nn.Module:
    """
    Instantiate a model from a config dict (the ``model`` section of a YAML).

    Required key
    ────────────
    type : str  — one of the keys in MODEL_REGISTRY

    Optional keys (with defaults from reference/config.py)
    ─────────────────────────────────────────────────────
    backbone    : "gru" | "lstm" | "transformer"  (default: "gru")
    hidden_dim  : int  (default: HIDDEN_DIM)
    num_layers  : int  (default: GRU_LAYERS)
    dropout     : float  (default: DROPOUT)
    gate_hidden : int  (default: 64, only used by moe)

    Raises
    ------
    ValueError if ``type`` is not in MODEL_REGISTRY.
    """
    mtype = model_cfg.get("type", "moe")
    if mtype not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{mtype}'. "
            f"Choose from: {sorted(MODEL_REGISTRY)}"
        )

    backbone   = model_cfg.get("backbone",   "gru")
    hidden_dim = model_cfg.get("hidden_dim", HIDDEN_DIM)
    num_layers = model_cfg.get("num_layers", GRU_LAYERS)
    dropout    = model_cfg.get("dropout",    DROPOUT)
    gate_h     = model_cfg.get("gate_hidden", 64)

    if mtype == "moe":
        return MoEFusionClassifier(backbone, hidden_dim, num_layers, dropout, gate_h)

    if mtype == "concat":
        return ConcatFusionClassifier(backbone, hidden_dim, num_layers, dropout)

    if mtype == "audio_sync_only":
        return AudioSyncOnlyClassifier(backbone, hidden_dim, num_layers, dropout)

    if mtype == "visual_only":
        return VisualOnlyClassifier(hidden_dim, dropout)

    if mtype == "pure_audio":
        return PureAudioClassifier(backbone, hidden_dim, num_layers, dropout)

    if mtype == "pure_video":
        return PureVideoClassifier(backbone, hidden_dim, num_layers, dropout)

    raise ValueError(f"Unhandled model type: {mtype}")
