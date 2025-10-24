"""
RL-based routing module for early-exit decisions.

This module learns when to exit early vs continue to deeper layers,
balancing classification accuracy against computational cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RoutingModule(nn.Module):
    """
    Policy network that decides whether to exit at each layer.

    Uses a small neural network to map layer features + context to exit probability.
    Trained with policy gradient methods (REINFORCE) to maximize reward:
        reward = correct_prediction - cost_penalty * layer_depth

    Args:
        feature_dims: Mapping from exit layer index to feature dimension
        hidden_dim: Hidden layer size for routing network
        context_dim: Dimension of additional context (e.g., running statistics)
        temperature: Temperature for probability scaling
    """

    def __init__(
        self,
        feature_dims: Dict[int, int],
        hidden_dim: int = 64,
        context_dim: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims must provide at least one exit layer dimension")

        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.context_dim = context_dim
        self.exit_layers = sorted(feature_dims.keys())
        self.num_exits = len(self.exit_layers)

        # Embed actual layer index (supports non-contiguous exit layers)
        self.layer_embedding = nn.Embedding(max(self.exit_layers) + 1, hidden_dim)

        # Feature encoders per exit (handle varying feature dims)
        self.feature_encoders = nn.ModuleDict(
            {
                str(layer_idx): nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                )
                for layer_idx, dim in feature_dims.items()
            }
        )

        # Context encoder (for running cost/accuracy statistics)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        combined_dim = hidden_dim + (hidden_dim // 2) + hidden_dim

        # Policy head (outputs exit probability)
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        layer_idx: int,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute exit probability for current layer.

        Args:
            features: Features from backbone layer (batch_size, feature_dim)
            layer_idx: Current layer index
            context: Optional context vector (batch_size, context_dim)

        Returns:
            Exit probability (batch_size,) in range [0, 1]
        """
        batch_size = features.shape[0]

        # Encode features using the exit-specific encoder
        encoder_key = str(layer_idx)
        if encoder_key not in self.feature_encoders:
            raise ValueError(f"No feature encoder registered for exit layer {layer_idx}")
        feat_encoded = self.feature_encoders[encoder_key](features)

        # Encode layer position
        layer_tensor = torch.full(
            (batch_size,), layer_idx, device=features.device, dtype=torch.long
        )
        layer_encoded = self.layer_embedding(layer_tensor)

        # Encode context (or use zeros)
        if context is None:
            context = torch.zeros(
                batch_size, self.context_dim, device=features.device, dtype=features.dtype
            )
        elif context.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected context dimension {self.context_dim}, got {context.shape[-1]}"
            )
        ctx_encoded = self.context_encoder(context)

        # Concatenate all inputs
        combined = torch.cat([feat_encoded, layer_encoded, ctx_encoded], dim=-1)

        # Compute exit probability
        logits = self.policy_head(combined).squeeze(-1)
        exit_prob = torch.sigmoid(logits / self.temperature)

        return exit_prob

    def sample_action(
        self,
        features: torch.Tensor,
        layer_idx: int,
        context: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample exit decision from policy.

        Args:
            features: Features from backbone layer
            layer_idx: Current layer index
            context: Optional context vector

        Returns:
            Tuple of:
            - action: Binary exit decision (1=exit, 0=continue)
            - log_prob: Log probability of the action (for REINFORCE)
        """
        exit_prob = self.forward(features, layer_idx, context)

        # Sample from Bernoulli distribution
        dist = torch.distributions.Bernoulli(exit_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob


class AttentionRoutingModule(nn.Module):
    """
    Attention-based routing module using transformer architecture.

    This is an enhanced version that adapts the existing AttentionRouter
    for the RL-based early-exit framework.

    Args:
        feature_dims: Mapping from exit layer index to feature dimension
        hidden_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        temperature: Temperature for probability scaling
    """

    def __init__(
        self,
        feature_dims: Dict[int, int],
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims must provide at least one exit layer dimension")

        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.exit_layers = sorted(feature_dims.keys())
        self.num_exits = len(self.exit_layers)

        # Import transformer block from existing attention_router
        from models.attention_router import _TransformerBlock

        # Feature projections per exit
        self.feature_projs = nn.ModuleDict(
            {
                str(layer_idx): nn.Linear(dim, hidden_dim)
                for layer_idx, dim in feature_dims.items()
            }
        )

        # Layer embedding supports actual layer indices
        self.layer_embedding = nn.Embedding(max(self.exit_layers) + 1, hidden_dim)

        # Positional encoding for two tokens
        self.positional = nn.Parameter(torch.randn(1, 2, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [_TransformerBlock(hidden_dim, num_heads, dropout=0.1) for _ in range(num_layers)]
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        layer_idx: int,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute exit probability using attention mechanism.

        Args:
            features: Features from backbone layer (batch_size, feature_dim)
            layer_idx: Current layer index
            context: Optional context (unused in this version)

        Returns:
            Exit probability (batch_size,)
        """
        del context  # Unused in this variant

        batch_size = features.shape[0]
        proj_key = str(layer_idx)
        if proj_key not in self.feature_projs:
            raise ValueError(f"No feature projection registered for exit layer {layer_idx}")
        feature_proj = self.feature_projs[proj_key]

        # Project features (use same projection for interaction term)
        feat_primary = feature_proj(features)
        feat_interact = feature_proj(features**2)

        # Encode layer position
        layer_tensor = torch.full(
            (batch_size,), layer_idx, device=features.device, dtype=torch.long
        )
        layer_encoded = self.layer_embedding(layer_tensor)

        # Create two tokens: primary features + layer, interaction features + layer
        token1 = feat_primary + layer_encoded
        token2 = feat_interact + layer_encoded

        tokens = torch.stack([token1, token2], dim=1)
        tokens = tokens + self.positional

        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Pool and predict
        pooled = tokens.mean(dim=1)
        logits = self.policy_head(pooled).squeeze(-1)
        exit_prob = torch.sigmoid(logits / self.temperature)

        return exit_prob

    def sample_action(
        self,
        features: torch.Tensor,
        layer_idx: int,
        context: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample exit decision from attention-based policy.

        Returns:
            Tuple of (action, log_prob)
        """
        exit_prob = self.forward(features, layer_idx, context)

        dist = torch.distributions.Bernoulli(exit_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
