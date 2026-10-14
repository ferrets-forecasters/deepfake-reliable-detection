import pytest
import torch
from torch import nn

from deepfake_detection.training.fixed_gate import freeze_gate_at


class _Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, audio, visual):
        return self.gate(torch.stack([audio, visual], dim=-1)).squeeze(-1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = _Gate()


def test_gate_outputs_the_constant_for_any_input_and_stays_frozen():
    model = _Model()
    freeze_gate_at(model, 0.5)
    out = model.gate(torch.randn(6), torch.randn(6) * 50)
    assert torch.allclose(out, torch.full((6,), 0.5))
    assert not any(p.requires_grad for p in model.gate.parameters())


def test_other_constants_and_bad_values():
    model = _Model()
    freeze_gate_at(model, 0.2)
    assert torch.allclose(model.gate(torch.randn(3), torch.randn(3)), torch.full((3,), 0.2), atol=1e-6)
    with pytest.raises(ValueError):
        freeze_gate_at(_Model(), 1.0)
