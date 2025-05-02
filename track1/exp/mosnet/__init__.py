"modules."

from track1.core.config.dataset import DatasetConfig
from track1.core.config.loss import LossConfig
from track1.core.config.ml import MLConfig
from track1.core.config.model import ModelConfig
from track1.core.config.path import PathConfig


class Config:
    """Configuration class for the track3 project."""

    data: DatasetConfig = DatasetConfig()
    ml: MLConfig = MLConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    path: PathConfig = PathConfig()


__all__ = ["Config"]
