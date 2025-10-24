"""
RL-based routing module for early-exit decisions.

This module learns when to exit early vs continue to deeper layers,
balancing classification accuracy against computational cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RoutingModule(nn.Module):
    """
    Policy network that decides whether to exit at each layer.

    Uses a small neural network to map layer features + context to exit probability.
    Trained with policy gradient methods (REINFORCE) to maximize reward:
        reward = correct_prediction - cost_penalty * layer_depth

    Args:
        feature_dim: Dimension of input features from backbone layers
        hidden_dim: Hidden layer size for routing network
        num_exits: Total number of exit points
        context_dim: Dimension of additional context (e.g., running statistics)
        temperature: Temperature for probability scaling
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_exits: int = 3,
        context_dim: int = 8,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_exits = num_exits
        self.temperature = temperature

        # Embed layer index
        self.layer_embedding = nn.Embedding(num_exits, hidden_dim)

        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Context encoder (for running cost/accuracy statistics)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Policy head (outputs exit probability)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim, hidden_dim),
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

        # Encode features
        feat_encoded = self.feature_encoder(features)

        # Encode layer position
        layer_tensor = torch.tensor([layer_idx], device=features.device)
        layer_encoded = self.layer_embedding(layer_tensor)
        layer_encoded = layer_encoded.expand(batch_size, -1)

        # Encode context (or use zeros)
        if context is None:
            context = torch.zeros(batch_size, 8, device=features.device)
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
        feature_dim: Dimension of input features
        hidden_dim: Transformer hidden dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        num_exits: Total number of exit points
        temperature: Temperature for probability scaling
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        num_exits: int = 3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.num_exits = num_exits

        # Import transformer block from existing attention_router
        from models.attention_router import _TransformerBlock

        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)

        # Layer embedding
        self.layer_embedding = nn.Embedding(num_exits, hidden_dim)

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
        batch_size = features.shape[0]

        # Project features
        feat_primary = self.feature_proj(features)
        feat_interact = self.feature_proj(features ** 2)

        # Encode layer position
        layer_tensor = torch.tensor([layer_idx], device=features.device)
        layer_encoded = self.layer_embedding(layer_tensor)
        layer_encoded = layer_encoded.unsqueeze(0).expand(batch_size, -1)

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
