from __future__ import annotations

from typing import Callable

import torch


def pooled_gradient(score_fn: Callable[[torch.Tensor], torch.Tensor], pooled: torch.Tensor) -> torch.Tensor:
    """d score / d pooled for a classifier that scores one pooled encoder vector.
    score_fn must return a scalar (for example the fake logit minus the real logit)."""
    pooled = pooled.detach().clone().requires_grad_(True)
    score = score_fn(pooled)
    (grad,) = torch.autograd.grad(score, pooled)
    return grad


def token_cam(tokens: torch.Tensor, grad_pooled: torch.Tensor) -> torch.Tensor:
    """Exact class activation map for features pooled by a mean over tokens.
    pooled = mean_n tokens[n], so d score / d tokens[n] = grad_pooled / N and the
    gradient times activation of token n is tokens[n] . grad_pooled / N.
    tokens: [N, D], grad_pooled: [D]. Returns [N]; positive values push toward the scored class."""
    return tokens @ grad_pooled / tokens.shape[0]


def fold_video_tokens(cam: torch.Tensor, tubelet_steps: int, grid: int) -> torch.Tensor:
    """[tubelet_steps * grid * grid] token scores to [tubelet_steps, grid, grid]."""
    return cam.view(tubelet_steps, grid, grid)


def positive_part(cam: torch.Tensor) -> torch.Tensor:
    """ReLU and scale to [0, 1], the usual Grad-CAM display range."""
    cam = torch.relu(cam)
    peak = cam.max()
    return cam / peak if peak > 0 else cam
