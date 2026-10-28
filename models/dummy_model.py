import torch
import torch.nn as nn

from models.base_model import BaseModel


class DummyModel(BaseModel):
    """
    Dummy model for testing the framework.
    """

    def __init__(
        self,
        history=12,
        horizon=12,
        num_nodes=113,
        in_channels=11,
        hidden_dim=64,
    ):

        super().__init__(
            num_nodes=num_nodes,
            input_dim=in_channels,
            history=history,
            horizon=horizon,
        )

        self.encoder = nn.Sequential(
            nn.Linear(history * in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon)
        )

    def forward(self, x):

        self.validate_input(x)

        B, T, N, F = x.shape

        # (B,T,N,F) -> (B,N,T*F)
        x = x.reshape(B, N, T * F)

        # MLP
        x = self.encoder(x)

        # (B,N,H) -> (B,H,N)
        x = x.permute(0, 2, 1)

        return x