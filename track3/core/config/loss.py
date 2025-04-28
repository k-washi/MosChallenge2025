LOSS_LIST = [
    "l1",
]


class LossConfig:
    """Loss config class."""

    loss_name: str = "l1"
    loss_list: list = LOSS_LIST

    # beta
    l1_rate_min: float = 0.0
    l1_rate_max: float = 0.6
    cl_rate: float = 0.5
    rank_rate: float = 1.0
    diff_rate: float = 0.0
    l1_loss_margin: float = 0.25
    ranking_loss_margin: float = 0.0
    contrastive_loss_margin: float = 0.5
