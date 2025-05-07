import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.models import Conformer
from transformers import WavLMModel  # pyright: ignore[reportPrivateImportUsage]

from track3.core.models.modules.film import FiLM


class ConformerStack(nn.Module):
    """Conformer stack for hierarchical subsampling."""

    def __init__(self, d_model: int = 768, n_heads: int = 8) -> None:
        """Initialize the Conformer stack."""
        super().__init__()
        self.stage = nn.ModuleList()
        for stride in [2, 2, 1]:  # hierarchical subsampling
            self.stage.append(
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, stride=stride, padding=1, groups=d_model),
                    nn.GELU(),
                    Conformer(
                        input_dim=d_model,
                        num_layers=2,  # 2 blocks / stage
                        ffn_dim=4 * d_model,
                        num_heads=n_heads,
                        depthwise_conv_kernel_size=31,
                        dropout=0.1,
                    ),
                )
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Conformer stack."""
        # x: (B,T,H)   mask: (B,T)  True=keep, False=pad
        for mod in self.stage:
            # Conv1d expects (B,C,T)
            x = x.transpose(1, 2)
            x = mod[0](x)  # depth-wise conv (subsampling) # pyright: ignore[reportIndexIssue]
            x = x.transpose(1, 2)  # back to (B,T,H)
            # update mask to match subsampling stride
            stride = mod[0].stride[0]  # pyright: ignore[reportIndexIssue]
            mask = mask[:, ::stride]
            # Conformer blocks
            lengths = mask.sum(-1).to(x.device).long()
            x, _ = mod[2](x, lengths=lengths)  # pyright: ignore[reportIndexIssue]
        return x, mask  # (B,T',H), (B,T')


# ───────────────   Transformer Pooling Head   ─────────────── #
class TransformerPool(nn.Module):
    """Transformer Pooling Head for hierarchical subsampling."""

    def __init__(self, d_model: int = 768, n_heads: int = 8, n_layers: int = 2) -> None:
        """Initialize the Transformer Pooling Head."""
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer Pooling Head."""
        b = x.size(0)
        cls_tok = self.cls.expand(b, -1, -1)  # (B,1,H)
        x = torch.cat([cls_tok, x], dim=1)  # (B,1+T,H)
        ext_mask = torch.cat([torch.ones(b, 1, device=x.device, dtype=torch.bool), mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~ext_mask)
        return x[:, 0]


class Head(nn.Module):
    """A PyTorch model for predicting Mean Opinion Score (MOS) using BiLSTM with attention."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        ds_hidden_dim: int = 32,
        ds_num: int = 1,
    ) -> None:
        """Initialize the BiLSTM with attention model."""
        super().__init__()
        self.ds_embed = nn.Embedding(ds_num, ds_hidden_dim)
        self.film = FiLM(ds_hidden_dim, input_dim)
        self.conformer = ConformerStack(d_model=input_dim, n_heads=num_heads)
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.t_pool = TransformerPool(d_model=input_dim, n_heads=num_heads, n_layers=num_layers)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, ds_id: torch.Tensor, feat_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.film(x, self.ds_embed(ds_id))
        out, mask = self.conformer(x, feat_mask)  # (batch, time, hidden)
        out = self.t_pool(out, mask)
        out = self.post_lstm_dropout(out)
        # Attentionスコア計算
        out = F.gelu(self.fc1(out))
        out = self.post_lstm_dropout(out)
        return self.output(out).squeeze(-1)  # → スカラー出力


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
        pretrained_model_name: str = "microsoft/wavlm-base-plus",
        n_layers: int = 3,
        n_heads: int = 8,
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
            hidden_dim=hidden_size,
            num_layers=n_layers,
            num_heads=n_heads,
            dropout=dropout,
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
