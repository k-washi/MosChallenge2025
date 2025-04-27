import torch
import torch.nn.functional as F
from torch import nn


class BiLSTMWithAttention(nn.Module):
    """A PyTorch model for predicting Mean Opinion Score (MOS) using BiLSTM with attention."""

    def __init__(
        self, input_dim: int, lstm_hidden_dim: int = 256, lstm_layers: int = 2, lstm_dropout: float = 0.05, dropout: float = 0.1
    ) -> None:
        """Initialize the BiLSTM with attention model."""
        super().__init__()

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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the model."""
        lstm_out, _ = self.bilstm(x)  # → (batch, time, 2 * hidden)
        lstm_out = self.post_lstm_dropout(lstm_out)
        # Attentionスコア計算
        attn_scores = self.attn(lstm_out).squeeze(-1)  # (batch, time)
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch, time)

        # 重み付き平均
        pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  # (batch, 2*hidden)
        o = F.gelu(self.fc1(pooled))
        o = self.post_lstm_dropout(o)
        return self.output(o).squeeze(-1)  # → スカラー出力
