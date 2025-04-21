class OptimizerConfig:
    """Optimizer config class."""

    optimizer_name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8

    warmup_epoch: int = 1
    num_cycles: float = 0.5


class MLConfig:
    """ML config class."""

    batch_size: int = 32
    test_batch_size: int = 32
    num_workers: int = 4

    num_epochs: int = 10
    accumulate_grad_num: int = 1
    optimizer: OptimizerConfig = OptimizerConfig()
