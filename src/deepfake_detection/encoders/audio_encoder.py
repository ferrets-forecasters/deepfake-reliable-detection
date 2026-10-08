from __future__ import annotations

import torch
from torch import nn
from transformers import WhisperModel

from deepfake_detection.config import Config


class AudioEncoderOutput:
    def __init__(self, pooled: torch.Tensor, temporal_seq: torch.Tensor):
        self.pooled = pooled
        self.temporal_seq = temporal_seq


WHISPER_MEL_FRAMES_PER_SECOND = 100
WHISPER_ENCODER_DOWNSAMPLE = 2


class AudioEncoder(nn.Module):
    """Frozen Whisper-medium encoder. Only the encoder stack runs, the decoder is unused.

    Every segment is zero-padded to Whisper's fixed 3,000-mel-frame (30s) input
    window, but a 2-second segment only carries real signal in the first
    segment_seconds*100 mel frames. The encoder's last_hidden_state is trimmed
    to the corresponding real_encoder_steps before pooling or downstream use,
    so the padded silence never dilutes the representation.
    """

    def __init__(self, config: Config):
        super().__init__()
        whisper = WhisperModel.from_pretrained(config.encoders.audio.model_id)
        self.encoder = whisper.get_encoder()
        del whisper.decoder
        if config.encoders.audio.frozen:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            self.encoder.eval()
        real_mel_frames = config.preprocessing.segment_seconds * WHISPER_MEL_FRAMES_PER_SECOND
        self.real_encoder_steps = real_mel_frames // WHISPER_ENCODER_DOWNSAMPLE

    @torch.no_grad()
    def forward(self, input_features: torch.Tensor) -> AudioEncoderOutput:
        outputs = self.encoder(input_features=input_features)
        temporal_seq = outputs.last_hidden_state[:, : self.real_encoder_steps, :]
        pooled = temporal_seq.mean(dim=1)
        return AudioEncoderOutput(pooled=pooled, temporal_seq=temporal_seq)
