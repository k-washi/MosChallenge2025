"modules."

from track3.core.config.dataset import DatasetConfig
from track3.core.config.ml import MLConfig
class Config:
    """Configuration class for the track3 project."""
    data: DatasetConfig = DatasetConfig()
    ml: MLConfig = MLConfig()

__all__ = ["Config"]