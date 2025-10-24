"""
End-to-end training script for MIDAS early-exit network with RL routing.

Trains both the classification network and routing policy jointly.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import wandb

from models.netflow_network import NetflowNetwork
from models.routing_module import RoutingModule, AttentionRoutingModule
from utils.data import load_netflow_data, load_split_netflow_data
from utils.cost_model import get_cost_model
from utils.rl_utils import (
    AdvantageEstimator,
    REINFORCETrainer,
    compute_routing_reward,
    ExitStatistics,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, input_dim: int) -> tuple:
    """
    Create network and routing module from config.

    Returns:
        Tuple of (network, routing_module)
    """
    # Create main network
    network = NetflowNetwork(
        input_dim=input_dim,
        hidden_dims=config["model"]["hidden_dims"],
        exit_layers=config["model"]["exit_layers"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        exit_dropout=config["model"]["exit_dropout"],
    )

    # Create routing module
    routing_type = config["routing"]["type"]
    exit_feature_dims = {
        layer_idx: network.hidden_dims[layer_idx] for layer_idx in network.exit_layers
    }

    if routing_type == "mlp":
        routing_module = RoutingModule(
            feature_dims=exit_feature_dims,
            hidden_dim=config["routing"]["hidden_dim"],
            context_dim=config["routing"]["context_dim"],
            temperature=config["routing"]["temperature"],
        )
    elif routing_type == "attention":
        routing_module = AttentionRoutingModule(
            feature_dims=exit_feature_dims,
            hidden_dim=config["routing"]["hidden_dim"],
            num_heads=config["routing"]["num_heads"],
            num_layers=config["routing"]["num_layers"],
            temperature=config["routing"]["temperature"],
        )
    elif routing_type == "none":
        routing_module = None
    else:
        raise ValueError(f"Unknown routing type: {routing_type}")

    return network, routing_module


def train_epoch(
    network: nn.Module,
    routing_module: nn.Module | None,
    train_loader,
    optimizer_net,
    optimizer_routing,
    cost_model,
    config: dict,
    epoch: int,
    rl_trainer: REINFORCETrainer,
    device: str,
) -> dict:
    """
    Train for one epoch.

    Returns:
        Dictionary of training metrics
    """
    network.train()
    if routing_module is not None:
        routing_module.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_rl_loss = 0.0
    exit_stats = ExitStatistics(num_exits=len(config["model"]["exit_layers"]))

    cost_per_layer = config["cost"]["cost_per_layer"]
    cost_lambda = config["cost"]["lambda"]
    exit_loss_weights = config["training"]["exit_loss_weights"]
    warmup_epochs = config["training"]["rl"]["warmup_epochs"]

    use_routing = routing_module is not None and epoch >= warmup_epochs

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass through network (get all exits)
        final_logits, exit_data = network(features, return_all_exits=True)

        # Classification loss (weighted sum over all exits)
        cls_loss = 0.0
        exit_logits = {}

        for i, exit_idx in enumerate(network.exit_layers):
            if f"logits_{exit_idx}" in exit_data:
                logits = exit_data[f"logits_{exit_idx}"]
                exit_logits[exit_idx] = logits

                weight = exit_loss_weights[i] if i < len(exit_loss_weights) else 1.0
                cls_loss += weight * F.cross_entropy(logits, labels)

        # Normalize by number of exits
        cls_loss = cls_loss / len(network.exit_layers)

        # RL routing loss (only after warmup)
        rl_loss = 0.0

        if use_routing:
            log_probs = []
            exit_probs = []
            rewards = []

            # Simulate routing decisions
            for i, exit_idx in enumerate(network.exit_layers[:-1]):  # Skip final exit
                features_at_exit = exit_data[f"features_{exit_idx}"]
                logits_at_exit = exit_logits[exit_idx]

                # Get routing decision
                action, log_prob = routing_module.sample_action(
                    features_at_exit, exit_idx, context=None
                )

                # Compute reward for this exit
                reward = compute_routing_reward(
                    prediction=logits_at_exit,
                    target=labels,
                    exit_layer=exit_idx,
                    cost_per_layer=cost_per_layer,
                    cost_lambda=cost_lambda,
                )

                log_probs.append(log_prob)
                exit_probs.append(routing_module(features_at_exit, exit_idx, context=None))
                rewards.append(reward)

            # Compute RL loss
            if log_probs:
                rl_loss = rl_trainer.compute_loss(log_probs, rewards, exit_probs)

        # Total loss
        total_loss_batch = cls_loss + config["training"]["routing_loss_weight"] * rl_loss

        # Backward pass
        if optimizer_net is not None:
            optimizer_net.zero_grad()
        if optimizer_routing is not None:
            optimizer_routing.zero_grad()

        total_loss_batch.backward()

        # Gradient clipping
        if config["training"]["grad_clip"] > 0:
            if optimizer_net is not None:
                nn.utils.clip_grad_norm_(network.parameters(), config["training"]["grad_clip"])
            if routing_module is not None:
                nn.utils.clip_grad_norm_(
                    routing_module.parameters(), config["training"]["grad_clip"]
                )

        if optimizer_net is not None:
            optimizer_net.step()
        if optimizer_routing is not None and use_routing:
            optimizer_routing.step()

        # Update statistics
        total_loss += total_loss_batch.item()
        total_cls_loss += cls_loss.item()
        total_rl_loss += rl_loss.item() if isinstance(rl_loss, torch.Tensor) else rl_loss

        # Log to progress bar
        pbar.set_postfix(
            {
                "loss": f"{total_loss_batch.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "rl": f"{rl_loss.item():.4f}" if isinstance(rl_loss, torch.Tensor) else "0.0",
            }
        )

    # Compute average metrics
    num_batches = len(train_loader)
    metrics = {
        "train_loss": total_loss / num_batches,
        "train_cls_loss": total_cls_loss / num_batches,
        "train_rl_loss": total_rl_loss / num_batches,
    }

    return metrics


def validate(
    network: nn.Module,
    routing_module: nn.Module | None,
    val_loader,
    cost_model,
    config: dict,
    device: str,
) -> dict:
    """
    Validate the model.

    Returns:
        Dictionary of validation metrics
    """
    network.eval()
    if routing_module is not None:
        routing_module.eval()

    exit_stats = ExitStatistics(num_exits=len(config["model"]["exit_layers"]))
    cost_per_layer = config["cost"]["cost_per_layer"]
    max_cost = sum(cost_per_layer)

    total_samples = 0
    total_correct = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            if routing_module is None:
                # Oracle: use final exit only
                final_logits, _ = network(features, return_all_exits=False)
                predictions = final_logits.argmax(dim=-1)
                correct = (predictions == labels).float()

                exit_layer = network.exit_layers[-1]
                exit_idx = len(network.exit_layers) - 1  # Last exit index
                cost = sum(cost_per_layer[: exit_layer + 1])

                for i in range(len(labels)):
                    is_correct = bool(correct[i])
                    exit_stats.update(exit_idx, is_correct, cost)

            else:
                # RL routing
                final_logits, exit_data = network(features, return_all_exits=True)

                # For each sample, simulate routing decision
                for sample_idx in range(len(labels)):
                    exited = False

                    for i, exit_layer in enumerate(network.exit_layers[:-1]):
                        features_at_exit = exit_data[f"features_{exit_layer}"][sample_idx:sample_idx+1]
                        exit_prob = routing_module(features_at_exit, exit_layer, context=None)

                        # Threshold-based exit during eval
                        if exit_prob > 0.5:
                            logits_at_exit = exit_data[f"logits_{exit_layer}"][sample_idx:sample_idx+1]
                            prediction = logits_at_exit.argmax(dim=-1)
                            is_correct = bool(prediction == labels[sample_idx])
                            cost = sum(cost_per_layer[: exit_layer + 1])

                            exit_stats.update(i, is_correct, cost)  # Use exit index i, not layer
                            exited = True
                            break

                    if not exited:
                        # Use final exit
                        final_exit_layer = network.exit_layers[-1]
                        final_exit_idx = len(network.exit_layers) - 1
                        prediction = final_logits[sample_idx].argmax()
                        is_correct = bool(prediction == labels[sample_idx])
                        cost = sum(cost_per_layer[: final_exit_layer + 1])

                        exit_stats.update(final_exit_idx, is_correct, cost)

            total_samples += len(labels)

    # Get metrics
    metrics = exit_stats.get_metrics()
    # Add normalized average cost for clarity
    if metrics.get("avg_cost") is not None and max_cost > 0:
        metrics["avg_cost_normalized"] = metrics["avg_cost"] / max_cost
    metrics = {f"val_{k}": v for k, v in metrics.items()}

    return metrics


def save_checkpoint(
    network: nn.Module,
    routing_module: nn.Module | None,
    optimizer_net,
    optimizer_routing,
    epoch: int,
    metrics: dict,
    config: dict,
    checkpoint_path: str,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "network_state_dict": network.state_dict(),
        "optimizer_net_state_dict": optimizer_net.state_dict() if optimizer_net else None,
        "metrics": metrics,
        "config": config,
    }

    if routing_module is not None:
        checkpoint["routing_module_state_dict"] = routing_module.state_dict()

    if optimizer_routing is not None:
        checkpoint["optimizer_routing_state_dict"] = optimizer_routing.state_dict()

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    network: nn.Module,
    routing_module: nn.Module | None = None,
    optimizer_net = None,
    optimizer_routing = None,
    device: str = "cpu",
):
    """Load model checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    network.load_state_dict(checkpoint["network_state_dict"])

    if routing_module is not None and "routing_module_state_dict" in checkpoint:
        routing_module.load_state_dict(checkpoint["routing_module_state_dict"])

    if optimizer_net is not None and checkpoint.get("optimizer_net_state_dict"):
        optimizer_net.load_state_dict(checkpoint["optimizer_net_state_dict"])

    if optimizer_routing is not None and checkpoint.get("optimizer_routing_state_dict"):
        optimizer_routing.load_state_dict(checkpoint["optimizer_routing_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})

    print(f"Loaded checkpoint from epoch {epoch}")
    if metrics:
        print(f"Checkpoint metrics: {metrics}")

    return epoch, metrics


def main():
    parser = argparse.ArgumentParser(description="Train MIDAS early-exit network")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    device = config["training"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    elif device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        print("MPS not available, using CPU")

    print(f"Using device: {device}")

    # Load data
    if "train_path" in config["data"]:
        train_loader, val_loader, test_loader, scaler = load_split_netflow_data(
            train_path=config["data"]["train_path"],
            val_path=config["data"]["val_path"],
            test_path=config["data"]["test_path"],
            target_column=config["data"]["target_column"],
            normalize=config["data"]["normalize"],
            batch_size=config["data"]["batch_size"],
        )
    else:
        train_loader, val_loader, test_loader, scaler = load_netflow_data(
            data_path=config["data"]["data_path"],
            target_column=config["data"]["target_column"],
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            normalize=config["data"]["normalize"],
            random_state=config["data"]["random_seed"],
        )

    # Get input dimension from data
    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[1]
    print(f"Input dimension: {input_dim}")

    # Create model
    network, routing_module = create_model(config, input_dim)
    network = network.to(device)
    if routing_module is not None:
        routing_module = routing_module.to(device)

    # Load checkpoint if specified
    start_epoch = 0
    if "load_checkpoint" in config["training"] and config["training"]["load_checkpoint"]:
        checkpoint_path = config["training"]["load_checkpoint"]
        if Path(checkpoint_path).exists():
            start_epoch, _ = load_checkpoint(
                checkpoint_path=checkpoint_path,
                network=network,
                routing_module=routing_module,
                device=device,
            )
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}, training from scratch")

    # Freeze backbone if specified
    freeze_backbone = config["training"].get("freeze_backbone", False)
    if freeze_backbone:
        print("\n" + "="*80)
        print("FREEZING BACKBONE NETWORK")
        print("="*80)
        for name, param in network.named_parameters():
            param.requires_grad = False
            print(f"  Frozen: {name}")
        print(f"Total frozen parameters: {sum(p.numel() for p in network.parameters())}")
        print("="*80 + "\n")

    # Create optimizers
    # Only create optimizer for network if not frozen
    if freeze_backbone or config["training"]["lr_classifier"] == 0:
        optimizer_net = None
        print("Network optimizer: DISABLED (backbone frozen)")
    else:
        optimizer_net = torch.optim.AdamW(
            network.parameters(),
            lr=config["training"]["lr_classifier"],
            weight_decay=config["training"]["weight_decay"],
        )
        print(f"Network optimizer: AdamW (lr={config['training']['lr_classifier']})")

    optimizer_routing = None
    if routing_module is not None and config["training"]["lr_routing"] > 0:
        optimizer_routing = torch.optim.AdamW(
            routing_module.parameters(),
            lr=config["training"]["lr_routing"],
            weight_decay=config["training"]["weight_decay"],
        )
        trainable_params = sum(p.numel() for p in routing_module.parameters() if p.requires_grad)
        print(f"Routing optimizer: AdamW (lr={config['training']['lr_routing']}, params={trainable_params:,})")

    # Create cost model
    cost_model = get_cost_model(config["cost"])

    # Print cost profile for transparency
    cost_per_layer = config["cost"].get("cost_per_layer")
    if cost_per_layer:
        cumulative = []
        total = 0.0
        for c in cost_per_layer:
            total += c
            cumulative.append(total)
        print("\nCost profile (per-layer and cumulative):")
        print(f"  per-layer:   {cost_per_layer}")
        print(f"  cumulative:  {cumulative}")
        print(f"  max_cost:    {sum(cost_per_layer):.4f}")

    # Optional: print a FLOPs profile derived from architecture
    if config["cost"].get("print_flops_profile", False):
        layer_dims = [input_dim] + network.hidden_dims
        flops_per_layer = []
        cum_flops = []
        t = 0
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            fl = 2 * in_dim * out_dim
            flops_per_layer.append(fl)
            t += fl
            cum_flops.append(t)
        print("\nFLOPs profile (approx FC only):")
        print(f"  per-layer:   {flops_per_layer}")
        print(f"  cumulative:  {cum_flops}")
        print(f"  total FLOPs: {t:,}\n")

    # Create RL trainer
    advantage_estimator = AdvantageEstimator(beta=config["training"]["rl"]["advantage_beta"])
    rl_trainer = REINFORCETrainer(
        advantage_estimator=advantage_estimator,
        entropy_coef=config["training"]["rl"]["entropy_coef"],
    )

    # Initialize W&B
    if config["wandb"]["enabled"]:
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=config["wandb"]["run_name"],
            tags=config["wandb"]["tags"],
            config=config,
        )

    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    best_val_metric = 0.0
    patience_counter = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            network=network,
            routing_module=routing_module,
            train_loader=train_loader,
            optimizer_net=optimizer_net,
            optimizer_routing=optimizer_routing,
            cost_model=cost_model,
            config=config,
            epoch=epoch,
            rl_trainer=rl_trainer,
            device=device,
        )

        # Validate
        val_metrics = validate(
            network=network,
            routing_module=routing_module,
            val_loader=val_loader,
            cost_model=cost_model,
            config=config,
            device=device,
        )

        # Combine metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch}

        # Print metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics.get('val_overall_accuracy', 0):.4f}")
        if 'val_avg_cost' in val_metrics:
            if 'val_avg_cost_normalized' in val_metrics:
                print(
                    f"  Val Avg Cost: {val_metrics['val_avg_cost']:.4f} (norm={val_metrics['val_avg_cost_normalized']:.4f})"
                )
            else:
                print(f"  Val Avg Cost: {val_metrics['val_avg_cost']:.4f}")

        # Log to W&B
        if config["wandb"]["enabled"]:
            wandb.log(metrics)

        # Early stopping
        if config["training"]["early_stopping"]["enabled"]:
            metric_name = config["training"]["early_stopping"]["metric"]
            current_metric = val_metrics.get(f"val_{metric_name.replace('val_', '')}", 0)

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0

                # Save best model (avoid overwriting baseline by default)
                checkpoint_name = config["training"].get("checkpoint_name")
                if not checkpoint_name:
                    checkpoint_name = (
                        "routing_best.pt" if routing_module is not None else "baseline_best.pt"
                    )
                best_checkpoint_path = checkpoint_dir / checkpoint_name
                save_checkpoint(
                    network=network,
                    routing_module=routing_module,
                    optimizer_net=optimizer_net,
                    optimizer_routing=optimizer_routing,
                    epoch=epoch,
                    metrics=metrics,
                    config=config,
                    checkpoint_path=str(best_checkpoint_path),
                )
                print(f"  New best {metric_name}: {current_metric:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= config["training"]["early_stopping"]["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Finish W&B
    if config["wandb"]["enabled"]:
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
