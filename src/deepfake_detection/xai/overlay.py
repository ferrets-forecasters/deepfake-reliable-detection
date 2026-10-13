from __future__ import annotations

import cv2
import numpy as np


def normalize_map(heat: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a flat map stays zero."""
    heat = heat.astype(np.float32)
    span = heat.max() - heat.min()
    return (heat - heat.min()) / span if span > 1e-12 else np.zeros_like(heat)


def resize_map(heat: np.ndarray, height: int, width: int) -> np.ndarray:
    return np.clip(cv2.resize(heat.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC), 0.0, None)


def overlay_heatmap(image_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] heat map (any size) over an RGB uint8 image."""
    height, width = image_rgb.shape[:2]
    heat = normalize_map(resize_map(heat, height, width))
    colored = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * image_rgb + alpha * colored, 0, 255).astype(np.uint8)


def overlay_video_cam(frames_rgb: list[np.ndarray], cam: np.ndarray, tubelet: int = 2, alpha: float = 0.45) -> list[np.ndarray]:
    """cam is [time_steps, grid, grid]; each time step covers `tubelet` consecutive frames.
    All steps share one scale so brightness is comparable across frames."""
    peak = float(cam.max())
    scaled = cam / peak if peak > 0 else cam
    return [overlay_heatmap(frame, scaled[min(i // tubelet, len(scaled) - 1)], alpha) for i, frame in enumerate(frames_rgb)]


def box_to_crop(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """A box in frame coordinates expressed as fractions of another box (for example a mouth box inside the face crop)."""
    width, height = outer[2] - outer[0], outer[3] - outer[1]
    return (
        float(np.clip((inner[0] - outer[0]) / width, 0, 1)), float(np.clip((inner[1] - outer[1]) / height, 0, 1)),
        float(np.clip((inner[2] - outer[0]) / width, 0, 1)), float(np.clip((inner[3] - outer[1]) / height, 0, 1)),
    )


def overlay_cam_on_box(frame_rgb: np.ndarray, box: tuple[float, float, float, float], heat: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Draws a [0, 1] heat map on the face box of a full frame. The map is the CAM of the squashed
    224x224 face crop, so it is stretched back to the box. Blend weight follows the heat, so cold
    areas keep the original pixels."""
    height, width = frame_rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame_rgb
    heat = np.clip(resize_map(heat, y1 - y0, x1 - x0), 0.0, 1.0)
    colored = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    weight = (strength * heat)[..., None]
    out = frame_rgb.copy()
    out[y0:y1, x0:x1] = np.clip((1 - weight) * out[y0:y1, x0:x1] + weight * colored, 0, 255).astype(np.uint8)
    return out
