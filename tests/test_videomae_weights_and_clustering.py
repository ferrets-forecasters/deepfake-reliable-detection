import numpy as np
import torch
from transformers import VideoMAEConfig, VideoMAEModel

from deepfake_detection.encoders.videomae_weights import restore_attention_biases
from deepfake_detection.inference.diarization import EcapaDiarizer


def tiny_model() -> VideoMAEModel:
    config = VideoMAEConfig(image_size=32, patch_size=16, num_frames=2, tubelet_size=2, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32)
    return VideoMAEModel(config)


def test_q_and_v_biases_are_copied_into_query_and_value():
    model = tiny_model()
    for layer in model.encoder.layer:
        layer.attention.attention.query.bias.data.zero_()
        layer.attention.attention.value.bias.data.zero_()
    state = {}
    for index in range(2):
        state[f"encoder.layer.{index}.attention.attention.q_bias"] = torch.full((16,), 0.5 + index)
        state[f"videomae.encoder.layer.{index}.attention.attention.v_bias"] = torch.full((16,), -0.25)
    assert restore_attention_biases(model, state) == 2
    first = model.encoder.layer[0].attention.attention
    assert torch.allclose(first.query.bias, torch.full((16,), 0.5))
    assert torch.allclose(model.encoder.layer[1].attention.attention.query.bias, torch.full((16,), 1.5))
    assert torch.allclose(first.value.bias, torch.full((16,), -0.25))
    assert float(first.key.bias.abs().sum()) == 0.0


def clusterer(distance: float = 0.8) -> EcapaDiarizer:
    diarizer = object.__new__(EcapaDiarizer)
    diarizer.same_speaker_distance = distance
    return diarizer


def two_voices() -> np.ndarray:
    rng = np.random.default_rng(0)
    a, b = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])
    return np.stack([a + 0.1 * rng.normal(size=4) for _ in range(6)] + [b + 0.1 * rng.normal(size=4) for _ in range(6)])


def test_clustering_splits_two_voices_and_keeps_one_voice_together():
    embeddings = two_voices()
    assert len(set(clusterer()._cluster(embeddings, 2).tolist())) == 2
    assert len(set(clusterer()._cluster(embeddings[:6], 2).tolist())) == 1


def test_speaker_count_is_capped_by_the_number_of_faces():
    rng = np.random.default_rng(1)
    three = np.stack([np.eye(4)[i % 3] + 0.05 * rng.normal(size=4) for i in range(9)])
    assert len(set(clusterer()._cluster(three, 3).tolist())) == 3
    assert len(set(clusterer()._cluster(three, 2).tolist())) == 2
    assert len(set(clusterer()._cluster(three, 1).tolist())) == 1
