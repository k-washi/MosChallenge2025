import torch
from torch import nn
from transformers import Wav2Vec2Model  # pyright: ignore[reportPrivateImportUsage]


class MOSPredictorW2V2(nn.Module):
    """A PyTorch model for predicting Mean Opinion Score (MOS) using Wav2Vec2."""

    def __init__(
        self, ssl_out_dim: int = 768, dropout: float = 0.1, pretrained_model_name: str = "facebook/wav2vec2-base-960h"
    ) -> None:
        """Initialize the MOSPredictorW2V model."""
        super().__init__()

        self.ssl_model = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.ssl_out_dim = ssl_out_dim
        self.dense = nn.Linear(in_features=self.ssl_out_dim, out_features=self.ssl_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.acticvation = nn.Tanh()

        self.output_layer = nn.Linear(in_features=self.ssl_out_dim, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, 1).

        """
        # Extract features using Wav2Vec2 # ssl_out_dim = 768 (wav2vec2-base-960h)
        ssl_out = self.ssl_model(x).last_hidden_state  # (batch_size, seq_len, ssl_out_dim)
        # Take the mean of the features across the time dimension
        ssl_out = ssl_out.mean(dim=1)  # (batch_size, ssl_out_dim)
        # Pass through the output layer

        ssl_out = self.dense(ssl_out)  # (batch_size, ssl_out_dim)
        ssl_out = self.dropout(ssl_out)
        ssl_out = self.acticvation(ssl_out)
        # Pass through the output layer
        out = self.output_layer(ssl_out)  # (batch_size, 1)

        return out

    def freaze_ssl(self) -> None:
        """Freeze the SSL model parameters."""
        self.ssl_model.feature_extractor._freeze_parameters()  # noqa: SLF001


if __name__ == "__main__":
    # Example usage
    model = MOSPredictorW2V2()
    x = torch.randn(2, 320 * 6 + 159)  # Batch of 32 audio samples, each 1 second long
    output = model(x)
