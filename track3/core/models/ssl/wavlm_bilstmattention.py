import torch
from torch import nn
from transformers import WavLMModel  # pyright: ignore[reportPrivateImportUsage]

from track3.core.models.modules.bilstmattention import BiLSTMWithAttention


class MOSPredictorWavLM(nn.Module):
    """A PyTorch model for predicting Mean Opinion Score (MOS) using Wav2Vec2.

    [Audio Input]
     ↓
    [Wav2Vec2 / WavLM] → 出力: [batch, time, hidden]
        ↓
    [BiLSTM] → [batch, time, 2 * hidden_lstm]
        ↓
    [Attention Pooling] → [batch, pooled_hidden]
        ↓
    [Linear] → [batch, 1]（スカラー出力）
    """

    def __init__(
        self,
        dropout: float = 0.3,
        lstm_dropout: float = 0.05,
        pretrained_model_name: str = "microsoft/wavlm-base-plus",
        lstm_layrs: int = 3,
        lstm_hidden_dim: int = 256,
    ) -> None:
        """Initialize the MOSPredictorW2V model."""
        super().__init__()

        self.ssl_model = WavLMModel.from_pretrained(pretrained_model_name)
        hidden_size = self.ssl_model.config.hidden_size
        self.head = BiLSTMWithAttention(
            input_dim=hidden_size,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layrs,
            dropout=dropout,
            lstm_dropout=lstm_dropout,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, seq_len). Defaults to None.

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, 1).

        """
        # Extract features using Wav2Vec2 # ssl_out_dim = 768 (wav2vec2-base-960h)
        if attention_mask is None:
            ssl_out = self.ssl_model(x).last_hidden_state  # (batch_size, seq_len, ssl_out_dim)
        else:
            ssl_out = self.ssl_model(x, attention_mask=attention_mask).last_hidden_state  # (batch_size, seq_len, ssl_out_dim)
        if attention_mask is not None:
            # Apply attention mask to the output
            output_seq_len = ssl_out.size(1)
            attention_mask = (
                torch.nn.functional.interpolate(
                    attention_mask.unsqueeze(1).float(),
                    size=output_seq_len,
                    mode="nearest",
                )
                .squeeze(1)
                .to(torch.bool)
            )
        return self.head(ssl_out, attention_mask=attention_mask)  # (batch_size, 1)

    def freaze_ssl(self) -> None:
        """Freeze the SSL model parameters."""
        self.ssl_model.feature_extractor._freeze_parameters()  # noqa: SLF001


if __name__ == "__main__":
    # Example usage
    model = MOSPredictorWavLM()
    x = torch.randn(2, 320 * 6 + 159)  # Batch of 32 audio samples, each 1 second long
    output = model(x)
    print(output.shape)  # Should print torch.Size([2, 1])
