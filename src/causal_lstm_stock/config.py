from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    ticker: str
    date_column: str
    lookback_window: int
    target_horizon_days: int
    train_split: float
    val_split: float
    paths: dict[str, str]
    modalities: dict[str, Any] = field(default_factory=dict)
    finbert: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    architecture: str
    input_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    num_classes: int


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    seed: int
    checkpoint_dir: str


@dataclass
class AppConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: str | Path) -> AppConfig:
    config_dir = Path(config_dir)
    data_raw = _read_yaml(config_dir / "data.yaml")["data"]
    model_raw = _read_yaml(config_dir / "model.yaml")["model"]
    train_raw = _read_yaml(config_dir / "train.yaml")["train"]

    return AppConfig(
        data=DataConfig(**data_raw),
        model=ModelConfig(**model_raw),
        train=TrainConfig(**train_raw),
    )
