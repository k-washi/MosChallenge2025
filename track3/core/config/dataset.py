DEFAULT_DATASET_MAP = {"track3": 0, "bvccmain": 1, "somos": 2}


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
    aug_rate: float = 0.5
    pitch_shift_max: int = 150
    time_wrap_min: float = 0.95
    time_wrap_max: float = 1.05

    # dataset extend
    # contrastiveのデータに対してmosスコアが付いているデータが少ないので拡張する
    is_extend: bool = False
    extend_rate: float = 0.1  # mosがついているデータをextendする割合

    train_dataset_dict: dict[str, int] = DEFAULT_DATASET_MAP
    test_dataset_dict: dict[str, int] = DEFAULT_DATASET_MAP
