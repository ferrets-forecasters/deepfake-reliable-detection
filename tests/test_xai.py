import numpy as np
import torch

from deepfake_detection.xai import (
    fold_video_tokens, input_gradient_saliency, occlusion_over_time, pooled_gradient,
    positive_part, shift_sweep, shift_waveform, token_cam,
)


def test_token_cam_sums_to_the_score_of_a_linear_head():
    torch.manual_seed(0)
    tokens = torch.randn(12, 8)
    head = torch.randn(8)
    score_fn = lambda pooled: pooled @ head
    grad = pooled_gradient(score_fn, tokens.mean(dim=0))
    cam = token_cam(tokens, grad)
    assert torch.allclose(cam.sum(), score_fn(tokens.mean(dim=0)), atol=1e-5)


def test_fold_and_positive_part():
    cam = torch.arange(2 * 3 * 3, dtype=torch.float32) - 9.0
    folded = fold_video_tokens(cam, 2, 3)
    assert folded.shape == (2, 3, 3)
    shown = positive_part(folded)
    assert shown.min() >= 0 and shown.max() == 1.0


def test_input_gradient_saliency_finds_the_used_cells():
    weights = torch.zeros(4, 6)
    weights[1, 2] = 3.0
    saliency = input_gradient_saliency(lambda x: (x * weights).sum(), torch.randn(4, 6))
    assert saliency[1, 2] == 3.0 and saliency.sum() == 3.0


def test_occlusion_marks_the_window_that_carries_the_score():
    signal = torch.zeros(2, 20)
    signal[:, 10:14] = 1.0
    drops = occlusion_over_time(lambda x: x.sum(), signal, time_dim=1, window=4, stride=4)
    best = max(drops, key=lambda row: row[2])
    assert (best[0], best[1]) == (8, 12) or (best[0], best[1]) == (12, 16)
    assert best[2] > 0


def test_shift_waveform_delays_and_advances():
    wave = np.arange(1, 6, dtype=np.float32)
    assert shift_waveform(wave, 2).tolist() == [0, 0, 1, 2, 3]
    assert shift_waveform(wave, -2).tolist() == [3, 4, 5, 0, 0]


def test_shift_sweep_reports_each_offset():
    wave = np.ones(1600, dtype=np.float32)
    rows = shift_sweep(lambda w: float(w.sum()), wave, 16000, [0, 50, -50])
    assert [ms for ms, _ in rows] == [0, 50, -50]
    assert rows[0][1] == 1600.0 and rows[1][1] == 800.0
