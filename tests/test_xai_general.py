import numpy as np
import pytest
import torch
from torch import nn

from deepfake_detection.xai import (
    GradCAM, box_to_crop, integrated_gradients, normalize_map, occlusion_2d, overlay_heatmap, overlay_video_cam, smoothgrad,
)


class TinyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)).mean(dim=(2, 3)))


def test_gradcam_on_a_conv_layer_gives_a_nonnegative_spatial_map():
    torch.manual_seed(0)
    model = TinyCnn()
    cam = GradCAM(model, model.conv)
    heat = cam(torch.randn(1, 1, 8, 8), lambda m, x: m(x).sum())
    cam.remove()
    assert heat.shape == (1, 8, 8) and heat.min() >= 0


def test_gradcam_on_tokens_peaks_at_the_token_that_carries_the_score():
    layer = nn.Linear(4, 4, bias=False)
    nn.init.eye_(layer.weight)
    model = nn.Sequential(layer)
    cam = GradCAM(model, layer)
    tokens = torch.zeros(1, 5, 4)
    tokens[0, 3] = 1.0
    heat = cam(tokens, lambda m, x: (m(x)[0, 3]).sum())
    cam.remove()
    assert heat.shape == (1, 5) and int(heat.argmax()) == 3


def test_integrated_gradients_sum_to_the_score_difference():
    x = torch.tensor([1.0, 2.0, -1.0])
    score = lambda v: (v ** 2).sum()
    attribution = integrated_gradients(score, x, steps=64)
    assert attribution.sum().item() == pytest.approx(score(x).item(), rel=1e-3)


def test_smoothgrad_of_a_linear_score_is_its_weight():
    weights = torch.tensor([1.0, -2.0, 0.5])
    grad = smoothgrad(lambda v: (v * weights).sum(), torch.randn(3), samples=4)
    assert torch.allclose(grad, weights)


def test_occlusion_2d_marks_the_patch_that_holds_the_signal():
    x = torch.zeros(8, 8)
    x[2:4, 4:6] = 1.0
    drops = occlusion_2d(lambda v: v.sum(), x, patch=2, stride=2)
    row, col = np.unravel_index(int(drops.argmax()), drops.shape)
    assert (row, col) == (1, 2) and drops.max() == 4.0


def test_overlay_keeps_the_image_shape_and_normalize_map_spans_zero_to_one():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    shown = overlay_heatmap(image, np.random.rand(4, 4))
    assert shown.shape == image.shape and shown.dtype == np.uint8
    scaled = normalize_map(np.array([[2.0, 4.0], [6.0, 10.0]]))
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert normalize_map(np.ones((2, 2))).max() == 0.0


def test_video_overlay_returns_one_image_per_frame():
    frames = [np.zeros((16, 16, 3), dtype=np.uint8)] * 6
    shown = overlay_video_cam(frames, np.random.rand(3, 4, 4))
    assert len(shown) == 6 and shown[0].shape == (16, 16, 3)


def test_box_to_crop_gives_fractions_of_the_outer_box():
    assert box_to_crop((20, 60, 40, 80), (0, 0, 100, 100)) == (0.2, 0.6, 0.4, 0.8)
    assert box_to_crop((120, 0, 140, 10), (0, 0, 100, 100))[0] == 1.0
