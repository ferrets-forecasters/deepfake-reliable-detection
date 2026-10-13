from __future__ import annotations

import cv2
import numpy as np
import torch

VIDEOMAE_ID = "MCG-NJU/videomae-base"
WHISPER_SAMPLE_RATE = 16000
WHISPER_MEL_FRAMES = 3000
VIDEOMAE_FRAMES = 16
CROP_SIZE = 224


class SegmentEmbedder:
    """The frozen-encoder features the exported checkpoints were trained on, for 2-second
    segments: VideoMAE-base mean over all tokens (768) and Whisper-medium encoder mean over
    all 1,500 steps of the zero-padded 30 s window (1,024). The audio pooling keeps the
    padded steps on purpose, because that is what the checkpoints saw. Features are computed
    in batches; with half=True the encoders run under fp16 autocast on the GPU."""

    def __init__(self, whisper_name: str = "medium", device: str | None = None, half: bool = True):
        import whisper
        from transformers import VideoMAEImageProcessor

        from deepfake_detection.encoders.videomae_weights import load_videomae

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.half = half and self.device.type == "cuda"
        self._whisper = whisper
        self.processor = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_ID)
        self.video_model = load_videomae(VIDEOMAE_ID).to(self.device).eval()
        self.audio_model = whisper.load_model(whisper_name, device=self.device).eval()

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.half)

    @torch.no_grad()
    def video_features(self, segments: list[list[np.ndarray]], batch_size: int = 8) -> np.ndarray:
        """segments: for each segment, 16 RGB crops of one face. Each crop is squashed to 224x224 first, as in the original extraction. Returns [N, 768]."""
        out = []
        for start in range(0, len(segments), batch_size):
            batch = [[cv2.resize(frame, (CROP_SIZE, CROP_SIZE)) for frame in crops] for crops in segments[start:start + batch_size]]
            inputs = self.processor(batch, return_tensors="pt").to(self.device)
            with self._autocast():
                hidden = self.video_model(**inputs).last_hidden_state
            out.append(hidden.mean(dim=1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), dtype=np.float32)

    def video_feature(self, face_frames: list[np.ndarray]) -> np.ndarray:
        return self.video_features([face_frames])[0]

    def log_mel(self, waveform: np.ndarray) -> torch.Tensor:
        """waveform: 2 s of mono audio at 16 kHz. Returns the log-mel [80, 3000], zero-padded to Whisper's 30 s window."""
        mel = self._whisper.log_mel_spectrogram(waveform.astype(np.float32)).to(self.device)
        if mel.shape[-1] < WHISPER_MEL_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, WHISPER_MEL_FRAMES - mel.shape[-1]))
        return mel[..., :WHISPER_MEL_FRAMES]

    @torch.no_grad()
    def audio_features(self, waveforms: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """waveforms: 2 s of mono audio at 16 kHz each. Returns [N, 1024]."""
        out = []
        for start in range(0, len(waveforms), batch_size):
            mel = torch.stack([self.log_mel(w) for w in waveforms[start:start + batch_size]])
            with self._autocast():
                hidden = self.audio_model.encoder(mel)
            out.append(hidden.mean(dim=1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1024), dtype=np.float32)

    def audio_feature(self, waveform: np.ndarray) -> np.ndarray:
        return self.audio_features([waveform])[0]
