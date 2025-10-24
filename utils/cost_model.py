"""
Cost modeling utilities for early-exit networks.

Provides different cost computation methods based on:
- Layer depth (simple incremental)
- FLOPs (floating point operations)
- Latency (actual inference time)
- Energy (power consumption)
"""

import torch
import torch.nn as nn
from typing import List, Dict
import time


class CostModel:
    """
    Base class for cost computation.
    """

    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        """
        Compute cost of exiting at a specific layer.

        Args:
            exit_layer: Index of exit layer (0-indexed)
            total_layers: Total number of layers in network

        Returns:
            Cost value
        """
        raise NotImplementedError


class LayerDepthCost(CostModel):
    """
    Simple cost model based on layer depth.

    Cost increases linearly with depth, optionally with layer-specific weights.
    """

    def __init__(self, cost_per_layer: List[float] | None = None):
        """
        Args:
            cost_per_layer: Optional list of costs for each layer.
                          If None, uses uniform cost of 1 per layer.
        """
        self.cost_per_layer = cost_per_layer

    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        """Compute cumulative cost up to exit layer."""
        if self.cost_per_layer is not None:
            return sum(self.cost_per_layer[: exit_layer + 1])
        else:
            # Uniform cost
            return float(exit_layer + 1)


class ExponentialCost(CostModel):
    """
    Exponential cost model where deeper layers cost more.

    Useful for networks where layer size/complexity grows with depth.
    """

    def __init__(self, base: float = 1.5, offset: float = 1.0):
        """
        Args:
            base: Exponential base (cost = base^layer)
            offset: Additive offset
        """
        self.base = base
        self.offset = offset

    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        """Compute exponential cost."""
        cost = 0.0
        for i in range(exit_layer + 1):
            cost += self.offset + self.base ** i
        return cost


class FLOPsCost(CostModel):
    """
    Cost model based on floating point operations (FLOPs).

    Computes actual FLOPs for each layer based on layer dimensions.
    """

    def __init__(self, layer_dims: List[int]):
        """
        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ...]
        """
        self.layer_dims = layer_dims
        self.flops_per_layer = self._compute_flops_per_layer()

    def _compute_flops_per_layer(self) -> List[float]:
        """Compute FLOPs for each layer (FC layer: 2 * in_dim * out_dim)."""
        flops = []
        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]
            layer_flops = 2 * in_dim * out_dim  # Matrix multiply
            flops.append(layer_flops)
        return flops

    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        """Compute cumulative FLOPs up to exit layer."""
        return sum(self.flops_per_layer[: exit_layer + 1])


class LatencyCost(CostModel):
    """
    Cost model based on actual inference latency.

    Measures wall-clock time for forward pass through each layer.
    """

    def __init__(self, model: nn.Module, device: str = "cpu", num_warmup: int = 10):
        """
        Args:
            model: The neural network model
            device: Device to run profiling on
            num_warmup: Number of warmup iterations before profiling
        """
        self.model = model
        self.device = device
        self.latency_per_layer = self._profile_latency(num_warmup)

    def _profile_latency(self, num_warmup: int, num_runs: int = 100) -> List[float]:
        """Profile latency for each layer."""
        self.model.eval()
        latencies = []

        # Create dummy input (adjust size as needed)
        dummy_input = torch.randn(1, self.model.input_dim, device=self.device)

        with torch.no_grad():
            # Warmup
            for _ in range(num_warmup):
                _ = self.model(dummy_input)

            # Profile each layer
            features = dummy_input
            for layer in self.model.backbone_layers:
                start_time = time.perf_counter()
                for _ in range(num_runs):
                    features = layer(features)
                end_time = time.perf_counter()

                avg_latency = (end_time - start_time) / num_runs
                latencies.append(avg_latency)

        return latencies

    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        """Compute cumulative latency up to exit layer."""
        return sum(self.latency_per_layer[: exit_layer + 1])


class CompositeCost(CostModel):
    """
    Composite cost model combining multiple cost dimensions.

    Allows weighting different cost factors (FLOPs, latency, etc.)
    """

    def __init__(self, cost_models: Dict[str, CostModel], weights: Dict[str, float]):
        """
        Args:
            cost_models: Dictionary of named cost models
            weights: Weights for each cost model (same keys)
        """
        self.cost_models = cost_models
        self.weights = weights

        # Validate
        if set(cost_models.keys()) != set(weights.keys()):
            raise ValueError("cost_models and weights must have same keys")

    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        """Compute weighted sum of all cost models."""
        total_cost = 0.0
        for name, model in self.cost_models.items():
            cost = model.compute_cost(exit_layer, total_layers)
            total_cost += self.weights[name] * cost
        return total_cost


def get_cost_model(config: Dict) -> CostModel:
    """
    Factory function to create cost model from configuration.

    Args:
        config: Dictionary with cost model configuration

    Returns:
        Instantiated cost model

    Example config:
        {
            "type": "layer_depth",
            "cost_per_layer": [1, 2, 3, 4, 5]
        }
        or
        {
            "type": "exponential",
            "base": 1.5,
            "offset": 1.0
        }
    """
    cost_type = config.get("type", "layer_depth")

    if cost_type == "layer_depth":
        return LayerDepthCost(cost_per_layer=config.get("cost_per_layer"))

    elif cost_type == "exponential":
        return ExponentialCost(
            base=config.get("base", 1.5),
            offset=config.get("offset", 1.0),
        )

    elif cost_type == "flops":
        if "layer_dims" not in config:
            raise ValueError("FLOPs cost model requires 'layer_dims'")
        return FLOPsCost(layer_dims=config["layer_dims"])

    elif cost_type == "latency":
        if "model" not in config:
            raise ValueError("Latency cost model requires 'model'")
        return LatencyCost(
            model=config["model"],
            device=config.get("device", "cpu"),
            num_warmup=config.get("num_warmup", 10),
        )

    elif cost_type == "composite":
        if "models" not in config or "weights" not in config:
            raise ValueError("Composite cost model requires 'models' and 'weights'")

        # Recursively create sub-models
        cost_models = {
            name: get_cost_model(model_config) for name, model_config in config["models"].items()
        }
        return CompositeCost(cost_models=cost_models, weights=config["weights"])

    else:
        raise ValueError(f"Unknown cost model type: {cost_type}")


def normalize_costs(costs: List[float], max_cost: float | None = None) -> List[float]:
    """
    Normalize costs to [0, 1] range.

    Args:
        costs: List of cost values
        max_cost: Maximum cost for normalization (if None, uses max of costs)

    Returns:
        Normalized costs
    """
    if max_cost is None:
        max_cost = max(costs)

    if max_cost == 0:
        return [0.0] * len(costs)

    return [c / max_cost for c in costs]
