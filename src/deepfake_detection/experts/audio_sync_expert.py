from __future__ import annotations

import torch
from torch import nn

from deepfake_detection.config import Config


class AudioSyncExpertOutput:
    def __init__(self, logit: torch.Tensor, anomaly: torch.Tensor):
        self.logit = logit
        self.anomaly = anomaly


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.scorer = nn.Linear(input_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(sequence), dim=1)
        return (sequence * weights).sum(dim=1)


class AudioSyncExpert(nn.Module):
    """Audio-branch anomaly expert.

    Audio tokens cross-attend to the visual temporal sequence; the residual
    between the attended and original audio representation is the audio-visual
    sync anomaly signal the gate consumes. The attended sequence then runs
    through a Transformer pre-encoder and a bidirectional GRU with attention
    pooling to produce the audio expert's classification logit.
    """

    def __init__(self, config: Config):
        super().__init__()
        d_model = config.model.gru_hidden
        self.audio_proj = nn.Linear(config.encoders.audio.hidden_size, d_model)
        self.visual_proj = nn.Linear(config.encoders.visual.hidden_size, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=config.model.cross_attn_heads, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=config.model.cross_attn_heads, batch_first=True
        )
        self.transformer_pre_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=config.model.gru_hidden,
            num_layers=config.model.gru_layers,
            dropout=config.model.gru_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.attention_pooling = AttentionPooling(config.model.gru_hidden * 2)
        self.classifier = nn.Linear(config.model.gru_hidden * 2, 1)

    def forward(self, audio_seq: torch.Tensor, visual_seq: torch.Tensor) -> AudioSyncExpertOutput:
        audio_q = self.audio_proj(audio_seq)
        visual_kv = self.visual_proj(visual_seq)
        attended, _ = self.cross_attn(query=audio_q, key=visual_kv, value=visual_kv)
        residual = attended - audio_q
        anomaly = residual.norm(dim=-1).mean(dim=1)

        fused = audio_q + attended
        encoded = self.transformer_pre_encoder(fused)
        gru_out, _ = self.gru(encoded)
        pooled = self.attention_pooling(gru_out)
        logit = self.classifier(pooled).squeeze(-1)

        return AudioSyncExpertOutput(logit=logit, anomaly=anomaly)
