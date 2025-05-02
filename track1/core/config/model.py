class ClapConfig:
    """CLAP config class."""

    pretrained_model_name: str = "laion/clap-htsat-fused"
    sample_rate: int = 48000
    output_dim = 512


class CNN14Config:
    """CNN14 config class."""

    pretrained_model_name: str = "pretrained/Cnn14_16k_mAP=0.438.pth"
    output_dim = 557


class MosNetModelConfig:
    """Model config class."""

    input_dim: int = 512
    aux_dim: int = 557
    hidden_dim: int = 1024
    tf_n_head: int = 16
    tf_n_layers: int = 2
    tf_dropout: float = 0.1
    head_dropout: float = 0.2


class ModelConfig:
    """Model config class."""

    model_name: str = "mosnet"
    clap: ClapConfig = ClapConfig()
    cnn14: CNN14Config = CNN14Config()
    mosnet: MosNetModelConfig = MosNetModelConfig()
