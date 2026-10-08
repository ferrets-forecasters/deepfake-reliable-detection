from __future__ import annotations

import torch
from torch import nn
from transformers import VideoMAEModel

from deepfake_detection.config import Config


class VisualEncoderOutput:
    def __init__(self, pooled: torch.Tensor, temporal_seq: torch.Tensor):
        self.pooled = pooled
        self.temporal_seq = temporal_seq


class VisualEncoder(nn.Module):
    """Frozen VideoMAE-base encoder.

    VideoMAEModel exposes no class token (use_mean_pooling=True in the base
    checkpoint), so the pooled representation the paper calls the "[CLS]
    token" is the standard mean-pooled representation VideoMAEForVideoClassification
    itself uses: the average over all space-time patch tokens. The temporal
    sequence pools only the spatial axis, keeping one 768-dim vector per
    tubelet step, for experts that need a time axis.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(config.encoders.visual.model_id)
        self.hidden_size = config.encoders.visual.hidden_size
        self.tubelet_steps = config.preprocessing.video.num_frames // self.model.config.tubelet_size
        num_patches_per_frame = (config.preprocessing.video.image_size // self.model.config.patch_size) ** 2
        self.spatial_tokens = num_patches_per_frame
        if config.encoders.visual.frozen:
            for param in self.model.parameters():
                param.requires_grad_(False)
            self.model.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> VisualEncoderOutput:
        outputs = self.model(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state
        batch_size = sequence_output.shape[0]
        sequence_output = sequence_output.view(
            batch_size, self.tubelet_steps, self.spatial_tokens, self.hidden_size
        )
        temporal_seq = sequence_output.mean(dim=2)
        pooled = temporal_seq.mean(dim=1)
        return VisualEncoderOutput(pooled=pooled, temporal_seq=temporal_seq)
