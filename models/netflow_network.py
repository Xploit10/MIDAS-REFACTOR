"""
Early-exit MLP network for netflow classification.

This network has multiple intermediate exits where predictions can be made,
allowing for adaptive computation based on input difficulty.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from models.exit_head import ExitHead


class NetflowNetwork(nn.Module):
    """
    Deep MLP with multiple exit points for netflow classification.

    The network consists of:
    1. A backbone MLP with configurable depth
    2. Exit heads at specified layers
    3. Optional batch normalization and dropout

    Args:
        input_dim: Number of input features (netflow features)
        hidden_dims: List of hidden layer dimensions (e.g., [128, 256, 512, 256, 128])
        exit_layers: Indices of layers that have exits (e.g., [1, 3, 4])
        num_classes: Number of output classes
        dropout: Dropout probability for backbone
        exit_dropout: Dropout probability for exit heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        exit_layers: List[int],
        num_classes: int = 2,
        dropout: float = 0.2,
        exit_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.exit_layers = sorted(exit_layers)
        self.num_classes = num_classes
        self.num_exits = len(exit_layers)

        # Validate exit layers
        if max(exit_layers) >= len(hidden_dims):
            raise ValueError(f"Exit layer {max(exit_layers)} exceeds network depth {len(hidden_dims)}")

        # Build backbone layers
        self.backbone_layers = nn.ModuleList()
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.Sequential(
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.backbone_layers.append(layer)
            prev_dim = hidden_dim

        # Build exit heads at specified layers
        self.exit_heads = nn.ModuleDict()
        for exit_idx in exit_layers:
            exit_dim = hidden_dims[exit_idx]
            self.exit_heads[str(exit_idx)] = ExitHead(
                input_dim=exit_dim,
                num_classes=num_classes,
                dropout=exit_dropout,
            )

        # Final exit (always at last layer)
        if len(hidden_dims) - 1 not in exit_layers:
            self.exit_heads[str(len(hidden_dims) - 1)] = ExitHead(
                input_dim=hidden_dims[-1],
                num_classes=num_classes,
                dropout=exit_dropout,
            )
            self.exit_layers = sorted(exit_layers + [len(hidden_dims) - 1])

    def forward(
        self, x: torch.Tensor, return_all_exits: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            return_all_exits: If True, compute predictions at all exits.
                            If False, only return intermediate features.

        Returns:
            Tuple of:
            - final_logits: Logits from the final exit (batch_size, num_classes)
            - exit_data: Dictionary containing:
                - 'features_{i}': Features at exit layer i
                - 'logits_{i}': Logits at exit layer i (if return_all_exits=True)
        """
        exit_data = {}
        features = x

        for layer_idx, layer in enumerate(self.backbone_layers):
            features = layer(features)

            # Store features and optionally compute predictions at exit points
            if layer_idx in self.exit_layers:
                exit_data[f"features_{layer_idx}"] = features

                if return_all_exits:
                    exit_head = self.exit_heads[str(layer_idx)]
                    logits = exit_head(features)
                    exit_data[f"logits_{layer_idx}"] = logits

        # Final exit always returns logits
        final_layer_idx = len(self.backbone_layers) - 1
        final_logits = self.exit_heads[str(final_layer_idx)](features)

        return final_logits, exit_data

    def forward_with_routing(
        self, x: torch.Tensor, routing_module: nn.Module
    ) -> Tuple[torch.Tensor, int, Dict[str, torch.Tensor]]:
        """
        Forward pass with dynamic routing decisions.

        The network evaluates each exit point and uses the routing module
        to decide whether to exit early or continue to the next layer.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            routing_module: Module that decides whether to exit at each layer

        Returns:
            Tuple of:
            - predictions: Final predictions from the chosen exit
            - exit_layer: Index of the layer where we exited
            - routing_info: Dictionary with routing decisions and probabilities
        """
        routing_info = {
            "exit_probs": [],
            "exit_decisions": [],
        }

        features = x

        for layer_idx, layer in enumerate(self.backbone_layers):
            features = layer(features)

            # Check if this is an exit point
            if layer_idx in self.exit_layers:
                # Get routing decision
                exit_prob = routing_module(features, layer_idx)
                routing_info["exit_probs"].append(exit_prob)

                # Decide whether to exit (during inference)
                if not self.training:
                    should_exit = exit_prob > 0.5  # Simple threshold
                    routing_info["exit_decisions"].append(should_exit)

                    if should_exit.any():
                        # Exit here
                        exit_head = self.exit_heads[str(layer_idx)]
                        predictions = exit_head(features)
                        return predictions, layer_idx, routing_info

        # If we reach here, use the final exit
        final_layer_idx = len(self.backbone_layers) - 1
        final_predictions = self.exit_heads[str(final_layer_idx)](features)

        return final_predictions, final_layer_idx, routing_info

    def get_exit_logits(self, features: torch.Tensor, exit_idx: int) -> torch.Tensor:
        """
        Get predictions from a specific exit head.

        Args:
            features: Features at the exit layer
            exit_idx: Index of the exit layer

        Returns:
            Logits from the specified exit
        """
        if exit_idx not in self.exit_layers:
            raise ValueError(f"Layer {exit_idx} is not an exit point")

        exit_head = self.exit_heads[str(exit_idx)]
        return exit_head(features)
