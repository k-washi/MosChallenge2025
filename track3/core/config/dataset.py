class DatasetConfig:
    """Configuration class for the dataset."""

    sample_rate: int = 16000
    max_duration: float = 15
    label_min: int = 1
    label_max: int = 5
    is_label_normalize: bool = False
    normalize_scale: float = 1.001

    train_dataset_num: int = -1
