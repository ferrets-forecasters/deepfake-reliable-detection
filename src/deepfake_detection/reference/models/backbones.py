"""
Pluggable sequence encoders used inside AudioSyncExpert (and ConcatFusionClassifier).

All backbones share the same call contract:

    out = backbone(x, mask, lengths)

where:
    x       : [B, T, input_dim]
    mask    : [B, T] bool  — True = valid token
    lengths : [B] int      — unpadded sequence lengths
    out     : [B, T, backbone.out_dim]

The backbone's ``out_dim`` attribute tells downstream layers (AttentionPool,
classification head) how wide the output is.

Available backbones
───────────────────
  gru         BiGRU with skip-connection residual + LayerNorm
  lstm        BiLSTM with skip-connection residual + LayerNorm
  transformer Multi-layer TransformerEncoderLayer stack (no recurrence)

Backbone choice is passed as a string to factory.build_model() via the YAML
experiment config; see ``reference/models/factory.py``.
"""
import torch
import torch.nn as nn

from deepfake_detection.reference.config import DROPOUT


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class SequenceBackbone(nn.Module):
    """Abstract base — enforces the shared interface."""
    out_dim: int

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# BiGRU
# ─────────────────────────────────────────────────────────────────────────────

class GRUEncoder(SequenceBackbone):
    """
    Bidirectional GRU with a linear skip-connection residual.

    out_dim = hidden_dim * 2  (concatenated forward + backward hidden states)
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.out_dim = hidden_dim * 2
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj_res = nn.Linear(input_dim, hidden_dim * 2)
        self.norm_res = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, mask, lengths):
        packed        = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.gru(packed)
        out, _        = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # Skip connection over the valid time steps
        res = self.proj_res(x[:, :out.shape[1]])
        return self.norm_res(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

class LSTMEncoder(SequenceBackbone):
    """
    Bidirectional LSTM with a linear skip-connection residual.

    Compared with GRU, LSTM adds an explicit cell state which can retain
    longer-range temporal context — potentially useful when fake artifacts
    appear only in a few distant segments of a long video.

    out_dim = hidden_dim * 2
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.out_dim = hidden_dim * 2
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj_res = nn.Linear(input_dim, hidden_dim * 2)
        self.norm_res = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, mask, lengths):
        packed         = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _  = self.lstm(packed)
        out, _         = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        res = self.proj_res(x[:, :out.shape[1]])
        return self.norm_res(out + res)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer (no recurrence)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(SequenceBackbone):
    """
    Stack of TransformerEncoderLayers (pre-norm).

    Compared to GRU/LSTM:
      + Global self-attention captures long-range dependencies without
        sequential bottleneck; better at attending to sparse artifacts.
      − More parameters; needs more data to generalise.
      − No built-in inductive bias for temporal ordering (but sin positional
        encoding is added here).

    out_dim = hidden_dim
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int = 4, dropout: float = DROPOUT):
        super().__init__()
        self.out_dim = hidden_dim
        self.proj    = nn.Linear(input_dim, hidden_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.layers  = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            for _ in range(num_layers)
        ])

    def _positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encoding."""
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(1).float()         # [T, 1]
        dim = torch.arange(0, D, 2, device=x.device).float()                # [D/2]
        pe  = torch.zeros(T, D, device=x.device)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (dim / D)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (dim / D)))
        return x + pe.unsqueeze(0)

    def forward(self, x, mask, lengths):
        x = self.pos_drop(self._positional_encoding(self.proj(x)))
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

BACKBONE_REGISTRY = {
    "gru":         GRUEncoder,
    "lstm":        LSTMEncoder,
    "transformer": TransformerEncoder,
}


def build_backbone(name: str, input_dim: int, hidden_dim: int,
                   num_layers: int, dropout: float) -> SequenceBackbone:
    """
    Instantiate a backbone by name.

    Parameters
    ----------
    name       : one of "gru" | "lstm" | "transformer"
    input_dim  : width of the input feature vectors fed into the backbone
    hidden_dim : hidden/embedding dimension inside the backbone
    num_layers : number of GRU/LSTM layers, or Transformer depth
    dropout    : dropout probability
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Choose from: {list(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name](input_dim, hidden_dim, num_layers, dropout)
