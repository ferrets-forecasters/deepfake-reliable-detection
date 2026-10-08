from __future__ import annotations

import torch
from torch import nn

from deepfake_detection.config import Config
from deepfake_detection.experts import AudioSyncExpert, VisualArtifactExpert
from deepfake_detection.gate import ModalityReliabilityGate
from deepfake_detection.modeling.domain_classifier import DomainClassifier


class MoEFusionOutput:
    def __init__(
        self,
        final_logit: torch.Tensor,
        alpha: torch.Tensor,
        visual_logit: torch.Tensor,
        audio_logit: torch.Tensor,
        domain_logits: torch.Tensor | None,
    ):
        self.final_logit = final_logit
        self.alpha = alpha
        self.visual_logit = visual_logit
        self.audio_logit = audio_logit
        self.domain_logits = domain_logits


class MoEFusionClassifier(nn.Module):
    """Gated mixture-of-experts fusion: alpha * visual_logit + (1 - alpha) * audio_logit."""

    def __init__(self, config: Config, num_domains: int | None = None):
        super().__init__()
        self.visual_expert = VisualArtifactExpert(config)
        self.audio_expert = AudioSyncExpert(config)
        self.gate = ModalityReliabilityGate(config)
        self.domain_classifier = (
            DomainClassifier(config, num_domains=num_domains, lambd=config.train.domain_adversarial_weight)
            if num_domains and num_domains > 1
            else None
        )

    def forward(
        self, visual_seq: torch.Tensor, audio_seq: torch.Tensor, concat_repr: torch.Tensor | None = None
    ) -> MoEFusionOutput:
        visual_out = self.visual_expert(visual_seq)
        audio_out = self.audio_expert(audio_seq, visual_seq)
        alpha = self.gate(audio_out.anomaly, visual_out.anomaly)
        final_logit = alpha * visual_out.logit + (1 - alpha) * audio_out.logit

        domain_logits = None
        if self.domain_classifier is not None:
            assert concat_repr is not None, "concat_repr is required when domain adaptation is enabled"
            domain_logits = self.domain_classifier(concat_repr)

        return MoEFusionOutput(
            final_logit=final_logit,
            alpha=alpha,
            visual_logit=visual_out.logit,
            audio_logit=audio_out.logit,
            domain_logits=domain_logits,
        )
