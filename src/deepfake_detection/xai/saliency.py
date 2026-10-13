from __future__ import annotations

from typing import Callable

import torch


def input_gradient_saliency(score_fn: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor) -> torch.Tensor:
    """Absolute gradient of a scalar score with respect to the input, for example
    the Whisper log-mel [80, frames] or the VideoMAE pixel values. Needs gradients
    to flow through the (frozen) encoder, so score_fn must not run under no_grad."""
    inputs = inputs.detach().clone().requires_grad_(True)
    score = score_fn(inputs)
    (grad,) = torch.autograd.grad(score, inputs)
    return grad.abs()


@torch.no_grad()
def occlusion_over_time(
    score_fn: Callable[[torch.Tensor], torch.Tensor], inputs: torch.Tensor, time_dim: int, window: int, stride: int, fill: float = 0.0
) -> list[tuple[int, int, float]]:
    """Score drop when a window of the time axis is replaced by fill. Returns
    (start, end, base score minus occluded score); a large positive value means the
    window supports the scored class. Works for log-mel (time axis last) and for
    frame stacks (time axis first)."""
    base = float(score_fn(inputs))
    length = inputs.shape[time_dim]
    results = []
    for start in range(0, max(length - window, 0) + 1, stride):
        end = min(start + window, length)
        occluded = inputs.clone()
        index = [slice(None)] * inputs.dim()
        index[time_dim] = slice(start, end)
        occluded[tuple(index)] = fill
        results.append((start, end, base - float(score_fn(occluded))))
    return results
