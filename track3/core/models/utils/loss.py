from torch import nn

from track3.core.config.loss import LossConfig


def get_loss(config: LossConfig, loss_name: str) -> nn.Module:
    """Get the loss function based on the loss name.

    Args:
    ----
        config (LossConfig): Configuration object containing loss parameters.
        loss_name (str): Name of the loss function to retrieve.

    Returns:
    -------
        callable: The requested loss function.

    """
    if loss_name not in config.loss_list:
        emsg = f"Loss {loss_name} not found in loss list."
        raise ValueError(emsg)

    if loss_name == "l1":
        loss = nn.L1Loss()
    else:
        emsg = f"Loss {loss_name} not found."
        raise ValueError(emsg)

    return loss
