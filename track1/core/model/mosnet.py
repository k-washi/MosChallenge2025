import torch
from torch import nn


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
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + aux_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, z_audio: torch.Tensor, z_prompt: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
        ----
            z_audio (torch.Tensor): Audio features of shape (batch_size, seq_len, input_dim).
            z_prompt (torch.Tensor): Text features of shape (batch_size, seq_len, input_dim).
            aux (torch.Tensor): Auxiliary features of shape (batch_size, aux_dim).

        Returns:
        -------
            torch.Tensor: Output tensor of shape (batch_size, 1).
            torch.Tensor: Attention weights of shape (batch_size, 1).

        """
        z = torch.cat([z_audio, z_prompt, (z_audio - z_prompt).abs(), (z_audio * z_prompt)], dim=-1)
        z = self.proj(z)  # (batch_size, 1024)
        h = self.backbone(z).squeeze(1)  # (batch_size, 1024)
        mos_ta = self.head_align(h).squeeze(-1)
        mos_mi = self.head(torch.cat([h, aux], dim=-1)).squeeze(-1)
        return mos_mi, mos_ta


if __name__ == "__main__":
    # Example usage
    model = MOSNet()
    audio_features = torch.randn(2, 512)  # Batch of 32, sequence length of 10, feature dimension of 512
    prompt_features = torch.randn(2, 512)  # Same shape as audio features
    aux_features = torch.randn(2, 557)  # Batch of 32, auxiliary feature dimension of 557

    mos_mi, mos_ta = model(audio_features, prompt_features, aux_features)
    print(f"MOS MI Shape: {mos_mi.shape}")
    print(f"MOS TA Shape: {mos_ta.shape}")
