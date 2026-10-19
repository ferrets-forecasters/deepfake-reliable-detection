from pathlib import Path

import pytest
import yaml

from deepfake_detection.reference import load_reference

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "training"


@pytest.mark.parametrize("config, parameters", [("moe_fusion.yaml", 7_075_272), ("concat.yaml", 3_883_525)])
def test_reference_models_build_with_the_parameter_counts_of_the_paper_checkpoints(config, parameters):
    reference = load_reference()
    model = reference.build_model(yaml.safe_load((CONFIGS / config).read_text(encoding="utf-8"))["model"])
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_every_training_config_builds_a_model():
    reference = load_reference()
    for path in sorted(CONFIGS.glob("*.yaml")):
        reference.build_model(yaml.safe_load(path.read_text(encoding="utf-8"))["model"])
