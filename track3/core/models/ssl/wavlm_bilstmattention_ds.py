import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import WavLMModel  # pyright: ignore[reportPrivateImportUsage]

from track3.core.models.modules.film import FiLM


class Head(nn.Module):
    """A PyTorch model for predicting Mean Opinion Score (MOS) using BiLSTM with attention."""

    def __init__(
        self,
        input_dim: int,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.05,
        dropout: float = 0.1,
        ds_hidden_dim: int = 32,
        ds_num: int = 1,
    ) -> None:
        """Initialize the BiLSTM with attention model."""
        super().__init__()
        self.ds_embed = nn.Embedding(ds_num, ds_hidden_dim)
        self.film = FiLM(ds_hidden_dim, input_dim)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(2 * lstm_hidden_dim, 1)
        self.fc1 = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.output = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x: torch.Tensor, ds_id: torch.Tensor, feat_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.film(x, self.ds_embed(ds_id))
        if feat_mask is not None:
            lengths = feat_mask.sum(-1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.bilstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.bilstm(x)  # → (batch, time, 2 * hidden)
        lstm_out = self.post_lstm_dropout(lstm_out)

        lstm_out, _ = self.bilstm(x)  # → (batch, time, 2 * hidden)
        lstm_out = self.post_lstm_dropout(lstm_out)
        # Attentionスコア計算
        attn_scores = self.attn(lstm_out).squeeze(-1)  # (batch, time)
        if feat_mask is not None:
            attn_scores = attn_scores.masked_fill(~feat_mask, -1e4)

        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, time)

        # 重み付き平均
        pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, 2*hidden)
        o = F.gelu(self.fc1(pooled))
        o = self.post_lstm_dropout(o)
        return self.output(o).squeeze(-1)  # → スカラー出力


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
        ds_hidden_dim: int = 32,
        ds_num: int = 1,
    ) -> None:
        """Initialize the MOSPredictorW2V model."""
        super().__init__()

        self.ssl_model = WavLMModel.from_pretrained(
            pretrained_model_name,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_size = self.ssl_model.config.hidden_size

        self.head = Head(
            input_dim=hidden_size,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layrs,
            dropout=dropout,
            lstm_dropout=lstm_dropout,
            ds_hidden_dim=ds_hidden_dim,
            ds_num=ds_num,
        )

    def forward(self, x: torch.Tensor, ds_id: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the model.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            attention_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, seq_len). Defaults to None.
            ds_id (torch.Tensor): Dataset ID tensor of shape (batch_size, 1).

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
            attention_mask = self.ssl_model._get_feature_vector_attention_mask(ssl_out.size(1), attention_mask).to(  # pyright: ignore[reportArgumentType] # noqa: SLF001
                dtype=torch.bool
            )
        return self.head(ssl_out, ds_id, feat_mask=attention_mask)

    def freaze_ssl(self) -> None:
        """Freeze the SSL model parameters."""
        self.ssl_model.feature_extractor._freeze_parameters()  # noqa: SLF001


if __name__ == "__main__":
    # Example usage
    model = MOSPredictorWavLM(ds_num=2)
    x = torch.randn(2, 320 * 6 + 159)  # Batch of 32 audio samples, each 1 second long
    ds_id = torch.Tensor([0, 1]).long()  # Example dataset IDs
    attention_mask = torch.ones(2, 320 * 6 + 159).bool()  # Example attention mask
    attention_mask[0, 100:] = 0  # Example of masking the first sequence
    attention_mask[1, 50:] = 0  # Example of masking the second sequence
    output = model(x, ds_id, attention_mask)
    print(output.shape)  # Should print torch.Size([2, 1])
