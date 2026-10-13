from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class GradCAM:
    """Grad-CAM for any layer of a model, through a forward hook.

    Convolutional layers give [B, H, W] maps: channel weights are the spatial mean of the
    gradient, as in the original method. Transformer layers (activations [B, N, D]) give one
    value per token: gradient times activation summed over the feature axis, positive part.
    Fold the tokens back into their frame and patch grid with fold_video_tokens.

    score_fn(model, inputs) must return a scalar, for example the fake logit minus the real logit."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        self._activation = out
        if out.requires_grad:
            out.register_hook(lambda grad: setattr(self, "_gradient", grad))

    def __call__(self, inputs: torch.Tensor, score_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            score = score_fn(self.model, inputs)
            score.backward()
        if self._gradient is None:
            raise RuntimeError("no gradient reached the target layer; check that the input or the layer takes part in the score")
        activation, gradient = self._activation.detach(), self._gradient
        if activation.dim() == 4:
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            return torch.relu((weights * activation).sum(dim=1))
        if activation.dim() == 3:
            return torch.relu((gradient * activation).sum(dim=-1))
        raise ValueError(f"unsupported activation shape {tuple(activation.shape)}")

    def remove(self) -> None:
        self._handle.remove()
