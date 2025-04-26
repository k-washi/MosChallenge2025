class OptimizerConfig:
    """Optimizer config class."""

    optimizer_name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8

    warmup_epoch: float = 1
    num_cycles: float = 0.5


class MLConfig:
    """ML config class."""

    seed: int = 3407
    batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 4
    grad_clip_val: float = 500
    check_val_every_n_epoch: int = 1
    check_val_every_n_steps: int = 10000
    mix_precision: str = "32"

    num_epochs: int = 10
    accumulate_grad_num: int = 1
    optimizer: OptimizerConfig = OptimizerConfig()
