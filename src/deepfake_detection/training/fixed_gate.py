from __future__ import annotations

import math

import torch


def freeze_gate_at(model: torch.nn.Module, alpha: float) -> None:
    """Turns the modality gate into a constant: the last layer gets zero weights and the bias logit(alpha), and every
    gate parameter is frozen. The architecture and the state dict keys stay the same, so the checkpoint loads in the
    standard model and scores with the same constant."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    last = model.gate.gate[-2]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(math.log(alpha / (1.0 - alpha)))
    for parameter in model.gate.parameters():
        parameter.requires_grad_(False)
