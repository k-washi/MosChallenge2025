import torch

from track3.core.config.model import ModelConfig


def get_model(config: ModelConfig, model_name: str) -> torch.nn.Module:
    """Get the model based on the model name.

    Args:
    ----
        config (ModelConfig): Configuration object containing model parameters.
        model_name (str): Name of the model to retrieve.

    Returns:
    -------
        torch.nn.Module: The requested model class.

    """
    if model_name not in config.model_list:
        emsg = f"Model {model_name} not found in model list."
        raise ValueError(emsg)

    if model_name == "wav2vec2_linear":
        from track3.core.models.ssl.wav2vec2_linear import MOSPredictorW2V2

        model = MOSPredictorW2V2(
            dropout=config.w2v2.dropout,
            pretrained_model_name=config.w2v2.pretrained_model_name,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    elif model_name == "wav2vec2_bilstmattn":
        from track3.core.models.ssl.wav2vec2_bilstmattn import MOSPredictorW2V2

        model = MOSPredictorW2V2(
            dropout=config.w2v2.dropout,
            pretrained_model_name=config.w2v2.pretrained_model_name,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    else:
        emsg = f"Model {model_name} not found."
        raise ValueError(emsg)

    return model
