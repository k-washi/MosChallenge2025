from typing import Any

from track1.core.config import Config


def get_model(config: Config) -> Any:
    """Get the model based on the configuration.

    Args:
    ----
        config (Config): Configuration object containing model parameters.

    Returns:
    -------
        Any: The model instance.

    """
    if config.model.model_name == "mosnet":
        from track1.core.model.mosnet import MOSNet

        model = MOSNet(
            input_dim=config.model.mosnet.input_dim,
            hidden_dim=config.model.mosnet.hidden_dim,
            aux_dim=config.model.mosnet.aux_dim,
            tf_n_head=config.model.mosnet.tf_n_head,
            tf_n_layers=config.model.mosnet.tf_n_layers,
            tf_dropout=config.model.mosnet.tf_dropout,
            head_dropout=config.model.mosnet.head_dropout,
        )

    else:
        emsg = f"Model {config.model.model_name} not supported."
        raise NotImplementedError(emsg)

    return model
