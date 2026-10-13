from __future__ import annotations

from typing import Callable

import torch


def _gradient(score_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(score_fn(x), x)
    return grad


def integrated_gradients(
    score_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, baseline: torch.Tensor | None = None, steps: int = 32
) -> torch.Tensor:
    """Integrated gradients (Sundararajan et al.) with the midpoint rule along the straight path
    from the baseline (zeros by default) to x. The attributions sum to score(x) - score(baseline)
    up to the integration error."""
    baseline = torch.zeros_like(x) if baseline is None else baseline
    total = torch.zeros_like(x)
    for k in range(steps):
        point = baseline + (k + 0.5) / steps * (x - baseline)
        total += _gradient(score_fn, point)
    return (x - baseline) * total / steps


def smoothgrad(
    score_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, noise: float = 0.1, samples: int = 16, seed: int = 0
) -> torch.Tensor:
    """Mean gradient over noisy copies of x. noise is the standard deviation as a fraction of the input range."""
    generator = torch.Generator(device=x.device).manual_seed(seed)
    sigma = noise * (x.max() - x.min())
    total = torch.zeros_like(x)
    for _ in range(samples):
        total += _gradient(score_fn, x + sigma * torch.randn(x.shape, generator=generator, device=x.device))
    return total / samples


@torch.no_grad()
def occlusion_2d(
    score_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, patch: int, stride: int, fill: float = 0.0
) -> torch.Tensor:
    """Score drop when a patch x patch square of the last two axes is replaced by fill.
    Returns a [rows, cols] map over the patch positions; positive means the patch supports the score."""
    height, width = x.shape[-2:]
    base = float(score_fn(x))
    rows = range(0, max(height - patch, 0) + 1, stride)
    cols = range(0, max(width - patch, 0) + 1, stride)
    drops = torch.zeros(len(rows), len(cols))
    for r, top in enumerate(rows):
        for c, left in enumerate(cols):
            occluded = x.clone()
            occluded[..., top:top + patch, left:left + patch] = fill
            drops[r, c] = base - float(score_fn(occluded))
    return drops
