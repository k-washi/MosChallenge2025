import torch
from torch import nn
from transformers import (
    AutoModel,  # pyright: ignore[reportPrivateImportUsage]
    WavLMModel,  # pyright: ignore[reportPrivateImportUsage]
)


class MOSNet(nn.Module):
    """A simple neural network model for Mean Opinion Score (MOS) prediction."""

    def __init__(
        self,
        input_dim: int = 512,
        aux_dim: int = 557,
        hidden_dim: int = 1024,
        tf_n_head: int = 16,
        tf_n_layers: int = 2,
        tf_dropout: float = 0.1,
        head_dropout: float = 0.2,
    ) -> None:
        """Initialize the MOSNet model.

        Args:
        ----
            input_dim (int): Dimension of the input features. Default is 512.
            output_dim (int): Dimension of the output score. Default is 1.
            hidden_dim (int): Dimension of the hidden layer. Default is 1024.
            aux_dim (int): Dimension of the auxiliary features. Default is 557.
            tf_n_head (int): Number of attention heads in the transformer. Default is 16.
            tf_n_layers (int): Number of transformer layers. Default is 2.
            tf_dropout (float): Dropout rate for the transformer layers. Default is 0.1.
            head_dropout (float): Dropout rate for the final output layer. Default is 0.2.

        """
        super().__init__()
        _ = aux_dim  # Unused variable, but kept for compatibility with the original code

        # -------- 1. Music SSL encoder (HTS-AT) ----------
        self.audio_encoder = WavLMModel.from_pretrained(
            "microsoft/wavlm-base-plus",
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        self.audio_proj = nn.Linear(self.audio_encoder.config.hidden_size, hidden_dim)
        self.text_encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)

        # -------- 2. Fusion Transformer --------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=tf_n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=tf_dropout,
            batch_first=True,
        )
        self.fusion_tf = nn.TransformerEncoder(enc_layer, num_layers=tf_n_layers)

        self.proj = nn.Linear(input_dim * 4, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=tf_n_head,
            dim_feedforward=hidden_dim * 2,
            dropout=tf_dropout,
        )
        self.backbone = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=tf_n_layers,
        )

        self.head_align = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(head_dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(
        self,
        z_audio: torch.Tensor,
        z_prompt: torch.Tensor,
        aux: torch.Tensor,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        text: str,
        text_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
        ----
            z_audio (torch.Tensor): Audio features of shape (batch_size, seq_len, input_dim).
            z_prompt (torch.Tensor): Text features of shape (batch_size, seq_len, input_dim).
            aux (torch.Tensor): Auxiliary features of shape (batch_size, aux_dim).
            wav (torch.Tensor): Audio waveform of shape (batch_size, seq_len).
            wav_len (torch.Tensor): Length of the audio waveform of shape (batch_size,).
            text (str): Text input.
            text_len (torch.Tensor): Length of the text input of shape (batch_size,).

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, 1).
            torch.Tensor: Attention weights of shape (batch_size, 1).

        """
        _ = aux  # Unused variable, but kept for compatibility with the original code
        aud_out = self.audio_encoder(wav, wav_len, output_hidden_states=False)
        aud_out = aud_out.last_hidden_state

        aud_out = self.audio_proj(aud_out)

        text_out = self.text_encoder(text, text_len, output_hidden_states=False)
        text_out = text_out.last_hidden_state

        text_out = self.text_proj(text_out)

        fusion_inp = torch.cat([aud_out, text_out], dim=1)
        pad_text = text_len == 0
        pad_audio = torch.zeros(aud_out.size(0), aud_out.size(1), dtype=torch.bool).to(aud_out.device)
        fuse_mask = torch.cat([pad_audio, pad_text], dim=1)
        fused = self.fusion_tf(fusion_inp, src_key_padding_mask=fuse_mask)  # torch.Size([2, 69, 1024])
        pooled = fused[:, 0]  # CLS token # torch.Size([2, 1024])

        z = torch.cat([z_audio, z_prompt, (z_audio - z_prompt).abs(), (z_audio * z_prompt)], dim=-1)
        z = self.proj(z)  # (batch_size, 1024)
        h = self.backbone(z).squeeze(1)  # (batch_size, 1024)
        h = torch.cat([h, pooled], dim=1)
        mos_ta = self.head_align(h).squeeze(-1)
        mos_mi = self.head(h).squeeze(-1)
        return mos_mi, mos_ta


if __name__ == "__main__":
    # Example usage
    model = MOSNet()
    audio_features = torch.randn(2, 512)  # Batch of 32, sequence length of 10, feature dimension of 512
    prompt_features = torch.randn(2, 512)  # Same shape as audio features
    aux_features = torch.randn(2, 557)  # Batch of 32, auxiliary feature dimension of 557
    wav = torch.randn(2, 16000)  # Example audio waveform
    wav_len = torch.ones_like(wav)
    text_ids = torch.randint(0, 1000, (2, 20))  # Example text IDs
    text_len = torch.ones_like(text_ids)
    mos_mi, mos_ta = model(audio_features, prompt_features, aux_features, wav, wav_len, text_ids, text_len)
    print(f"MOS MI Shape: {mos_mi.shape}")
    print(f"MOS TA Shape: {mos_ta.shape}")
