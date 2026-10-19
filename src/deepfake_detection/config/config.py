from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _project_root() -> Path:
    override = os.environ.get("DEEPFAKE_PROJECT_ROOT")
    if override:
        return Path(override).resolve()
    return REPO_ROOT.parents[1]


@dataclass
class PathConfig:
    datasets_root: Path
    models_root: Path
    inferences_root: Path
    metrics_root: Path
    analyses_root: Path
    xai_root: Path

    @property
    def manifests_dir(self) -> Path:
        return self.datasets_root / "manifests"

    @property
    def features_dir(self) -> Path:
        return self.datasets_root / "features"

    @property
    def raw_dir(self) -> Path:
        return self.datasets_root / "raw"

    def checkpoints_dir(self, model_name: str) -> Path:
        return self.models_root / model_name

    def logs_dir(self, run_name: str) -> Path:
        return self.metrics_root / run_name / "logs"

    def results_dir(self, run_name: str) -> Path:
        return self.metrics_root / run_name

    def predictions_dir(self, run_name: str) -> Path:
        return self.inferences_root / run_name

    def analysis_dir(self, name: str) -> Path:
        return self.analyses_root / name

    def xai_dir(self, name: str) -> Path:
        return self.xai_root / name


@dataclass
class DatasetLayout:
    raw: str
    embedding: str
    labels: str
    manifest: str


@dataclass
class DatasetSpec:
    name: str
    modality: str
    media_types: list[str]
    fake_type: str
    audio_available: bool
    subject_rule: str
    real: int
    fake: int
    segments: int
    source_url: str
    download_url: str
    access: str


@dataclass
class SplitConfig:
    train: float
    val: float
    test: float
    level: str


@dataclass
class VideoPreprocessConfig:
    num_frames: int
    image_size: int
    face_detector_fps: int


@dataclass
class AudioPreprocessConfig:
    sample_rate: int
    mel_frames: int


@dataclass
class PreprocessConfig:
    segment_seconds: int
    video: VideoPreprocessConfig
    audio: AudioPreprocessConfig


@dataclass
class EncoderSpec:
    model_id: str
    hidden_size: int
    frozen: bool


@dataclass
class EncoderConfig:
    visual: EncoderSpec
    audio: EncoderSpec


@dataclass
class ModelConfig:
    concat_dim: int
    gru_hidden: int
    gru_layers: int
    gru_dropout: float
    visual_expert_hidden: int
    gate_hidden: int
    cross_attn_heads: int
    num_classes: int


@dataclass
class TrainConfig:
    optimizer: str
    learning_rate: float
    scheduler: str
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    swa_start_epoch: int
    focal_gamma: float
    focal_alpha: float
    label_smoothing: float
    domain_adversarial_weight: float
    gate_routing_aux_weight: float


@dataclass
class Config:
    seed: int
    paths: PathConfig
    layout: DatasetLayout
    manifest_columns: list[str]
    datasets: list[DatasetSpec]
    split: SplitConfig
    preprocessing: PreprocessConfig
    encoders: EncoderConfig
    model: ModelConfig
    train: TrainConfig

    def dataset_names(self) -> list[str]:
        return [d.name for d in self.datasets]

    def dataset(self, name: str) -> DatasetSpec:
        for spec in self.datasets:
            if spec.name == name:
                return spec
        raise KeyError(f"unknown dataset '{name}', known: {self.dataset_names()}")

    def dataset_dir(self, name: str) -> Path:
        return self.paths.datasets_root / self.dataset(name).name

    def raw_dir(self, name: str) -> Path:
        return self.dataset_dir(name) / self.layout.raw

    def embedding_dir(self, name: str) -> Path:
        return self.dataset_dir(name) / self.layout.embedding

    def labels_dir(self, name: str) -> Path:
        return self.dataset_dir(name) / self.layout.labels

    def manifest_path(self, name: str) -> Path:
        return self.labels_dir(name) / self.layout.manifest


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = REPO_ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    root = _project_root()
    paths = PathConfig(
        datasets_root=root / raw["paths"]["datasets_root"],
        models_root=root / raw["paths"]["models_root"],
        inferences_root=root / raw["paths"]["inferences_root"],
        metrics_root=root / raw["paths"]["metrics_root"],
        analyses_root=root / raw["paths"]["analyses_root"],
        xai_root=root / raw["paths"]["xai_root"],
    )
    datasets_path = Path(path).parent / raw["datasets_config"]
    with open(datasets_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f)
    layout = DatasetLayout(**registry["layout"])
    datasets = [DatasetSpec(**d) for d in registry["datasets"]]
    split = SplitConfig(**raw["split"])
    preprocessing = PreprocessConfig(
        segment_seconds=raw["preprocessing"]["segment_seconds"],
        video=VideoPreprocessConfig(**raw["preprocessing"]["video"]),
        audio=AudioPreprocessConfig(**raw["preprocessing"]["audio"]),
    )
    encoders = EncoderConfig(
        visual=EncoderSpec(**raw["encoders"]["visual"]),
        audio=EncoderSpec(**raw["encoders"]["audio"]),
    )
    model = ModelConfig(**raw["model"])
    train = TrainConfig(**raw["train"])

    return Config(
        seed=raw["seed"],
        paths=paths,
        layout=layout,
        manifest_columns=registry["manifest"]["columns"],
        datasets=datasets,
        split=split,
        preprocessing=preprocessing,
        encoders=encoders,
        model=model,
        train=train,
    )
