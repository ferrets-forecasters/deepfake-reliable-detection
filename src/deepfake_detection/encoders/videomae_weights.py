from __future__ import annotations

import torch
from transformers import VideoMAEModel


def restore_attention_biases(model: VideoMAEModel, state: dict[str, torch.Tensor]) -> int:
    """VideoMAE checkpoints store the attention biases as q_bias and v_bias (the key bias is zero).
    transformers 5 loads them into nothing and leaves query.bias and value.bias at zero, which
    changes every feature. Copies them in and returns the number of layers restored."""
    restored = 0
    for index, layer in enumerate(model.encoder.layer):
        attention = layer.attention.attention
        for source, target in (("q_bias", attention.query.bias), ("v_bias", attention.value.bias)):
            key = next((k for k in (f"encoder.layer.{index}.attention.attention.{source}", f"videomae.encoder.layer.{index}.attention.attention.{source}") if k in state), None)
            if key is None or target is None:
                continue
            with torch.no_grad():
                target.copy_(state[key])
        restored += 1
    return restored


def load_videomae(model_id: str) -> VideoMAEModel:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    model = VideoMAEModel.from_pretrained(model_id)
    first = model.encoder.layer[0].attention.attention
    if first.query.bias is not None and bool((first.query.bias == 0).all()):
        restore_attention_biases(model, load_file(hf_hub_download(model_id, "model.safetensors")))
    return model
