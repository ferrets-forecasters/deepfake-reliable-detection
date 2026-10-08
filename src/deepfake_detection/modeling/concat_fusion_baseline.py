from __future__ import annotations

import torch
from torch import nn

from deepfake_detection.config import Config
from deepfake_detection.modeling.domain_classifier import DomainClassifier


class ConcatFusionOutput:
    def __init__(self, final_logit: torch.Tensor, domain_logits: torch.Tensor | None):
        self.final_logit = final_logit
        self.domain_logits = domain_logits


class ConcatFusionClassifier(nn.Module):
    """Static-fusion ablation baseline ("Fusi Statis").

    The 1,792-dim z-scored, concatenated visual+audio vector is fed straight
    into a two-layer bidirectional GRU as a length-1 sequence, no gate, no
    gate-routing auxiliary loss. Isolates the gate's contribution against the
    MoE model under an otherwise matched training schedule.
    """

    def __init__(self, config: Config, num_domains: int | None = None):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config.model.concat_dim,
            hidden_size=config.model.gru_hidden,
            num_layers=config.model.gru_layers,
            dropout=config.model.gru_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(config.model.gru_hidden * 2, 1)
        self.domain_classifier = (
            DomainClassifier(config, num_domains=num_domains, lambd=config.train.domain_adversarial_weight)
            if num_domains and num_domains > 1
            else None
        )

    def forward(self, concat_repr: torch.Tensor) -> ConcatFusionOutput:
        sequence = concat_repr.unsqueeze(1)
        gru_out, _ = self.gru(sequence)
        pooled = gru_out[:, -1, :]
        final_logit = self.classifier(pooled).squeeze(-1)

        domain_logits = None
        if self.domain_classifier is not None:
            domain_logits = self.domain_classifier(concat_repr)

        return ConcatFusionOutput(final_logit=final_logit, domain_logits=domain_logits)
