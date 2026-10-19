"""
Shared architectural primitives used by both experts and the MoE classifier.
"""
import torch
import torch.nn as nn

from deepfake_detection.reference.config import DROPOUT


class GradientReversalFn(torch.autograd.Function):
    """Gradient Reversal Layer for DANN domain-adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between audio and video sequences.

    Returns the residual-added, normalised outputs *and* the raw attention
    outputs (before the residual add) so that callers can compute
    anomaly residuals: residual_a = a_enc - a_raw_attn.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn_av = nn.MultiheadAttention(dim, num_heads=4, dropout=DROPOUT, batch_first=True)
        self.attn_va = nn.MultiheadAttention(dim, num_heads=4, dropout=DROPOUT, batch_first=True)
        self.norm_a  = nn.LayerNorm(dim)
        self.norm_v  = nn.LayerNorm(dim)

    def forward(self, a, v, key_padding_mask):
        """
        Args:
            a                : [B, T, D]  audio features
            v                : [B, T, D]  video features
            key_padding_mask : [B, T] bool  True = valid token

        Returns:
            a_out      [B, T, D]  normalised residual-attended audio
            v_out      [B, T, D]  normalised residual-attended video
            a_raw_attn [B, T, D]  raw cross-attention output for audio (before residual)
            v_raw_attn [B, T, D]  raw cross-attention output for video (before residual)
        """
        inv = ~key_padding_mask                                        # MHA expects True = ignore
        a_raw_attn, _ = self.attn_av(a, v, v, key_padding_mask=inv)   # audio queries video
        v_raw_attn, _ = self.attn_va(v, a, a, key_padding_mask=inv)   # video queries audio
        a_out = self.norm_a(a + a_raw_attn)
        v_out = self.norm_v(v + v_raw_attn)
        return a_out, v_out, a_raw_attn, v_raw_attn


class AttentionPooling(nn.Module):
    """Soft attention pooling over the time dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask):
        """
        Args:
            x    : [B, T, D]
            mask : [B, T] bool  True = valid token
        Returns:
            [B, D]
        """
        scores  = self.score(x).squeeze(-1)                   # [B, T]
        scores  = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)                        # [B, D]
