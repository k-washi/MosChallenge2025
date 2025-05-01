import torch
from torch import nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) module."""

    def __init__(self, ds_embed_dim: int, hidden: int) -> None:
        """Initialize the FiLM module."""
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ds_embed_dim, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden * 2))

    def forward(self, x: torch.Tensor, ds_embed: torch.Tensor) -> torch.Tensor:
        """Forward pass of the FiLM module."""
        gamma, beta = self.mlp(ds_embed).chunk(2, dim=-1)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)
