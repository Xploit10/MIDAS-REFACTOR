"""
Reinforcement learning utilities for policy gradient training.

Includes REINFORCE algorithm, advantage estimation, and reward computation.
"""

import torch
import numpy as np
from typing import List, Dict


class AdvantageEstimator:
    """
    Estimates advantages for policy gradient using baseline subtraction.

    Uses exponential moving average as baseline to reduce variance.
    """

    def __init__(self, beta: float = 0.99):
        """
        Args:
            beta: Exponential moving average decay factor
        """
        self.beta = beta
        self.baseline = 0.0
        self.initialized = False

    def update(self, reward: float) -> None:
        """Update the baseline with a new reward."""
        if not self.initialized:
            self.baseline = reward
            self.initialized = True
        else:
            self.baseline = self.beta * self.baseline + (1 - self.beta) * reward

    def compute_advantage(self, reward: float) -> float:
        """
        Compute advantage as reward minus baseline.

        Args:
            reward: Observed reward

        Returns:
            Advantage value
        """
        if not self.initialized:
            return reward
        return reward - self.baseline

    def reset(self) -> None:
        """Reset the baseline."""
        self.baseline = 0.0
        self.initialized = False


class REINFORCETrainer:
    """
    REINFORCE (policy gradient) trainer for routing policy.

    Collects trajectories (states, actions, rewards) and updates policy
    to maximize expected reward.
    """

    def __init__(
        self,
        advantage_estimator: AdvantageEstimator | None = None,
        entropy_coef: float = 0.01,
    ):
        """
        Args:
            advantage_estimator: Optional advantage estimator for variance reduction
            entropy_coef: Coefficient for entropy regularization (encourages exploration)
        """
        self.advantage_estimator = advantage_estimator or AdvantageEstimator()
        self.entropy_coef = entropy_coef

    def compute_loss(
        self,
        log_probs: List[torch.Tensor],
        rewards: List[float],
        exit_probs: List[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Compute REINFORCE loss for a batch of trajectories.

        Loss = -sum(log_prob * advantage) - entropy_bonus

        Args:
            log_probs: List of log probabilities for each action in trajectory
            rewards: List of rewards for each action
            exit_probs: Optional list of exit probabilities (for entropy regularization)

        Returns:
            Policy gradient loss
        """
        policy_losses = []
        entropy_losses = []

        for i, (log_prob, reward) in enumerate(zip(log_probs, rewards)):
            # Compute advantage
            advantage = self.advantage_estimator.compute_advantage(reward)

            # Policy gradient loss
            policy_loss = -log_prob.mean() * advantage
            policy_losses.append(policy_loss)

            # Entropy regularization (encourage exploration)
            if exit_probs is not None and i < len(exit_probs):
                prob = exit_probs[i]
                entropy = -prob * torch.log(prob + 1e-8) - (1 - prob) * torch.log(1 - prob + 1e-8)
                entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy
                entropy_losses.append(entropy_loss)

            # Update baseline
            self.advantage_estimator.update(reward)

        # Combine losses
        total_policy_loss = torch.stack(policy_losses).sum()
        total_entropy_loss = torch.stack(entropy_losses).sum() if entropy_losses else 0.0

        return total_policy_loss + self.entropy_coef * total_entropy_loss


def compute_routing_reward(
    prediction: torch.Tensor,
    target: torch.Tensor,
    exit_layer: int,
    cost_per_layer: List[float],
    cost_lambda: float = 0.1,
) -> float:
    """
    Compute reward for routing decision.

    Reward balances classification accuracy against computational cost:
        reward = correctness - cost_lambda * cumulative_cost

    Args:
        prediction: Model predictions (batch_size, num_classes)
        target: Ground truth labels (batch_size,)
        exit_layer: Index of layer where we exited
        cost_per_layer: Cost of each layer (e.g., [1, 2, 3, 4, 5])
        cost_lambda: Weight for cost penalty

    Returns:
        Scalar reward value
    """
    # Compute accuracy
    pred_labels = prediction.argmax(dim=-1)
    correct = (pred_labels == target).float()
    accuracy = correct.mean().item()

    # Compute cumulative cost up to exit layer
    cumulative_cost = sum(cost_per_layer[: exit_layer + 1])

    # Normalize cost by maximum possible cost
    max_cost = sum(cost_per_layer)
    normalized_cost = cumulative_cost / max_cost

    # Compute reward
    reward = accuracy - cost_lambda * normalized_cost

    return reward


def compute_exit_metrics(
    predictions: Dict[int, torch.Tensor],
    targets: torch.Tensor,
    exit_counts: Dict[int, int],
    cost_per_layer: List[float],
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-exit evaluation.

    Args:
        predictions: Dictionary mapping exit_layer -> predictions
        targets: Ground truth labels
        exit_counts: Dictionary mapping exit_layer -> count of samples
        cost_per_layer: Cost of each layer

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Per-exit accuracy
    for exit_layer, preds in predictions.items():
        if preds is not None and len(preds) > 0:
            pred_labels = preds.argmax(dim=-1)
            correct = (pred_labels == targets).float()
            accuracy = correct.mean().item()
            metrics[f"accuracy_exit_{exit_layer}"] = accuracy

    # Exit usage distribution
    total_samples = sum(exit_counts.values()) or 1
    for exit_layer, count in exit_counts.items():
        metrics[f"usage_exit_{exit_layer}"] = count / total_samples

    # Average computational cost
    avg_cost = 0.0
    for exit_layer, count in exit_counts.items():
        layer_cost = sum(cost_per_layer[: exit_layer + 1])
        avg_cost += layer_cost * (count / total_samples)

    max_cost = sum(cost_per_layer)
    metrics["avg_cost"] = avg_cost
    metrics["avg_cost_normalized"] = avg_cost / max_cost

    # Compute cost-accuracy tradeoff metrics
    overall_correct = 0
    for exit_layer, preds in predictions.items():
        if preds is not None and len(preds) > 0:
            pred_labels = preds.argmax(dim=-1)
            correct = (pred_labels == targets).float().sum().item()
            overall_correct += correct

    overall_accuracy = overall_correct / total_samples if total_samples > 0 else 0.0
    metrics["overall_accuracy"] = overall_accuracy

    # Accuracy per unit cost
    if avg_cost > 0:
        metrics["accuracy_per_cost"] = overall_accuracy / avg_cost
    else:
        metrics["accuracy_per_cost"] = 0.0

    return metrics


class ExitStatistics:
    """
    Tracks statistics about exit decisions during training/evaluation.
    """

    def __init__(self, num_exits: int):
        self.num_exits = num_exits
        self.reset()

    def reset(self) -> None:
        """Reset all statistics."""
        self.exit_counts = {i: 0 for i in range(self.num_exits)}
        self.exit_correct = {i: 0 for i in range(self.num_exits)}
        self.total_cost = 0.0
        self.total_samples = 0

    def update(
        self,
        exit_layer: int,
        is_correct: bool,
        cost: float,
    ) -> None:
        """
        Update statistics with a new sample.

        Args:
            exit_layer: Index of exit layer used
            is_correct: Whether prediction was correct
            cost: Computational cost incurred
        """
        self.exit_counts[exit_layer] += 1
        if is_correct:
            self.exit_correct[exit_layer] += 1
        self.total_cost += cost
        self.total_samples += 1

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        metrics = {}

        # Per-exit accuracy
        for i in range(self.num_exits):
            count = self.exit_counts[i]
            if count > 0:
                accuracy = self.exit_correct[i] / count
                metrics[f"accuracy_exit_{i}"] = accuracy
                metrics[f"usage_exit_{i}"] = count / self.total_samples
            else:
                metrics[f"accuracy_exit_{i}"] = 0.0
                metrics[f"usage_exit_{i}"] = 0.0

        # Overall metrics
        total_correct = sum(self.exit_correct.values())
        if self.total_samples > 0:
            metrics["overall_accuracy"] = total_correct / self.total_samples
            metrics["avg_cost"] = self.total_cost / self.total_samples
            if self.total_cost > 0:
                metrics["accuracy_per_cost"] = (total_correct / self.total_samples) / (
                    self.total_cost / self.total_samples
                )
        else:
            metrics["overall_accuracy"] = 0.0
            metrics["avg_cost"] = 0.0
            metrics["accuracy_per_cost"] = 0.0

        return metrics
