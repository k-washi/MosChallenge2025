class DatasetConfig:
    """Configuration class for the dataset."""

    sample_rate: int = 16000
    max_duration: float = 15
    label_min: float = 1
    label_max: float = 5
    label_norm_max: float = 1
    label_norm_min: float = -1
    is_label_normalize: bool = False
    normalize_scale: float = 1.001

    train_dataset_num: int = -1

    # aug
    pitch_shift_max: int = 300
    time_wrap_min: float = 0.9
    time_wrap_max: float = 1.1
