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
    elif model_name == "wav2vec2_bilstmattention":
        from track3.core.models.ssl.wav2vec2_bilstmattention import MOSPredictorW2V2

        model = MOSPredictorW2V2(
            dropout=config.w2v2.dropout,
            lstm_dropout=config.w2v2.lstm_dropout,
            pretrained_model_name=config.w2v2.pretrained_model_name,
            lstm_layrs=config.w2v2.lstm_layers,
            lstm_hidden_dim=config.w2v2.lstm_hidden_dim,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    elif model_name == "wavlm_bilstmattention":
        from track3.core.models.ssl.wavlm_bilstmattention import MOSPredictorWavLM

        model = MOSPredictorWavLM(
            dropout=config.w2v2.dropout,
            lstm_dropout=config.w2v2.lstm_dropout,
            lstm_layrs=config.w2v2.lstm_layers,
            lstm_hidden_dim=config.w2v2.lstm_hidden_dim,
            pretrained_model_name=config.wavlm_pretrained_model_name,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    elif model_name == "hubert_bilstmattention":
        from track3.core.models.ssl.hubert_bilstmattention import MOSPredictorHubert

        model = MOSPredictorHubert(
            dropout=config.w2v2.dropout,
            lstm_dropout=config.w2v2.lstm_dropout,
            lstm_layrs=config.w2v2.lstm_layers,
            lstm_hidden_dim=config.w2v2.lstm_hidden_dim,
            pretrained_model_name=config.hubert_pretrained_model_name,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    elif model_name == "contentvec_bilstmattention":
        from track3.core.models.ssl.contentvec_bilstmattention import MOSPredictorContentVec

        model = MOSPredictorContentVec(
            dropout=config.w2v2.dropout,
            lstm_dropout=config.w2v2.lstm_dropout,
            lstm_layrs=config.w2v2.lstm_layers,
            lstm_hidden_dim=config.w2v2.lstm_hidden_dim,
            pretrained_model_name=config.contentvector_pretrained_model_name,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    elif model_name == "wavlm_bilstmattention_ds":
        from track3.core.models.ssl.wavlm_bilstmattention_ds import MOSPredictorWavLM

        model = MOSPredictorWavLM(
            dropout=config.w2v2.dropout,
            lstm_dropout=config.w2v2.lstm_dropout,
            lstm_layrs=config.w2v2.lstm_layers,
            lstm_hidden_dim=config.w2v2.lstm_hidden_dim,
            ds_hidden_dim=config.w2v2.ds_hidden_dim,
            ds_num=config.w2v2.ds_num,
            pretrained_model_name=config.wavlm_pretrained_model_name,
        )
        if config.w2v2.is_freeze_ssl:
            model.freaze_ssl()
    else:
        emsg = f"Model {model_name} not found."
        raise ValueError(emsg)

    return model
