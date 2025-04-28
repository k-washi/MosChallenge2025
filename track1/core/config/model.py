class ClapConfig:
    """CLAP config class."""

    pretrained_model_name: str = "laion/clap-htsat-fused"


class ModelConfig:
    """Model config class."""

    clap: ClapConfig = ClapConfig()
