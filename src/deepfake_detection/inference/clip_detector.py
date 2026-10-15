from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def preprocess() -> transforms.Compose:
    """The CLIP image processor of openai/clip-vit-large-patch14 written out: bicubic resize of the short side to 224, centre crop, CLIP mean and std."""
    return transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])


def square_crop(rgb: np.ndarray, box: tuple[float, float, float, float]) -> Image.Image:
    """Square crop around the centre of `box`, as wide as its longer side, kept inside the frame."""
    x0, y0, x1, y1 = box
    side = int(min(max(x1 - x0, y1 - y0), rgb.shape[0], rgb.shape[1]))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - side / 2, 0, rgb.shape[1] - side))
    top = int(np.clip(cy - side / 2, 0, rgb.shape[0] - side))
    return Image.fromarray(rgb[top:top + side, left:left + side])


def face_crops(path: Path, detector, frames: int) -> tuple[list[Image.Image], int]:
    """Square crops of the largest face in `frames` evenly spaced frames; frames without a face are dropped."""
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, max(count - 1, 0), min(frames, max(count, 1))).round().astype(int).tolist())
    crops = []
    for index in range(count):
        if not cap.grab():
            break
        if index not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect(rgb)
        if not faces:
            continue
        crops.append(square_crop(rgb, max((f.box for f in faces), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))))
    cap.release()
    return crops, len(wanted)


class ClipDetector:
    """The published CLIP LN-tuning detector (Yermakov et al., TorchScript export) scoring face crops."""

    def __init__(self, weights: Path, device: str = "cuda:0", dtype: torch.dtype = torch.bfloat16):
        self.device, self.dtype = device, dtype
        self.model = torch.jit.load(str(weights), map_location=device).to(device).to(dtype).eval()
        self.transform = preprocess()

    def p_fake(self, crops: list[Image.Image]) -> list[float]:
        batch = torch.stack([self.transform(c) for c in crops]).to(self.device).to(self.dtype)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.model(batch).float().softmax(dim=1)[:, 1].tolist()
