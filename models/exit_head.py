"""
Exit head module for early-exit classification.

Each exit point in the network uses an instance of this head to produce
class predictions from intermediate layer features.
"""

import torch
import torch.nn as nn


class ExitHead(nn.Module):
    """
    Small MLP classifier head attached to intermediate network layers.

    Args:
        input_dim: Dimension of input features from backbone
        hidden_dim: Hidden layer size (optional, defaults to input_dim // 2)
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or input_dim // 2

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.head(x)
