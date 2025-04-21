MODEL_LIST = [
    "wav2vec2_linear",
]


class Wav2Vec2Config:
    """wav2vec2 config class."""

    ssl_out_dim: int = 768
    dropout: float = 0.1
    pretrained_model_name: str = "facebook/wav2vec2-base-960h"
    is_freaze_ssl: bool = False


class ModelConfig:
    """Model config class."""

    model_name: str = "wav2vec2_linear"
    model_list: list = MODEL_LIST

    w2v2: Wav2Vec2Config = Wav2Vec2Config()
