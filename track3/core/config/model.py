MODEL_LIST = [
    "wav2vec2_linear",
    "wav2vec2_bilstmattn",
    "wav2vec2_bilstmattention",
    "wavlm_bilstmattention",
    "wavlm_bilstmattention_ds",
    "hubert_bilstmattention",
    "contentvec_bilstmattention",
    "wavlm_conformer_ds",
]


class Wav2Vec2Config:
    """wav2vec2 config class."""

    ssl_out_dim: int = 768
    dropout: float = 0.1
    lstm_dropout: float = 0.05
    lstm_layers: int = 3
    pretrained_model_name: str = "facebook/wav2vec2-base-960h"
    is_freeze_ssl: bool = False
    lstm_hidden_dim: int = 256

    # ds
    ds_num = 3
    ds_hidden_dim = 32

    # conformer
    n_layers: int = 3
    n_heads: int = 8


class ModelConfig:
    """Model config class."""

    model_name: str = "wav2vec2_linear"
    model_list: list = MODEL_LIST

    w2v2: Wav2Vec2Config = Wav2Vec2Config()
    wavlm_pretrained_model_name: str = "microsoft/wavlm-base-plus"
    hubert_pretrained_model_name: str = "facebook/hubert-base-ls960"
    contentvector_pretrained_model_name: str = "lengyue233/content-vec-best"
