"""
Evaluation script for MIDAS early-exit network.

Evaluates trained model against oracle baselines and logs results to W&B.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

from models.netflow_network import NetflowNetwork
from models.routing_module import RoutingModule, AttentionRoutingModule
from utils.data import load_netflow_data, load_split_netflow_data
from utils.cost_model import get_cost_model
from utils.rl_utils import ExitStatistics


def load_model(checkpoint_path: str, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Get input dimension from checkpoint state dict
    net_sd = checkpoint.get("network_state_dict") or checkpoint.get("network")
    if net_sd is None:
        raise KeyError("Checkpoint missing network state dict")
    first_layer_key = "backbone_layers.0.0.weight"
    if first_layer_key not in net_sd:
        # Try alternate key for robustness
        first_layer_key = next(k for k in net_sd.keys() if k.endswith(".0.weight"))
    input_dim = net_sd[first_layer_key].shape[1]

    # Recreate network
    network = NetflowNetwork(
        input_dim=input_dim,
        hidden_dims=config["model"]["hidden_dims"],
        exit_layers=config["model"]["exit_layers"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        exit_dropout=config["model"]["exit_dropout"],
    )
    network.load_state_dict(net_sd)
    network = network.to(device)

    # Recreate routing module if present
    routing_module = None
    if checkpoint.get("routing_module_state_dict") is not None or checkpoint.get("routing_module") is not None:
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

        # Accept both new and legacy keys
        routing_sd = checkpoint.get("routing_module_state_dict") or checkpoint.get("routing_module")
        routing_module.load_state_dict(routing_sd)
        routing_module = routing_module.to(device)

    return network, routing_module, config


def evaluate_oracle_always_exit(
    network: nn.Module,
    test_loader,
    exit_layer: int,
    cost_per_layer: list,
    device: str,
) -> dict:
    """
    Oracle baseline: always use a specific exit.

    Args:
        network: The network
        test_loader: Test data loader
        exit_idx: Index of exit to always use
        cost_per_layer: Cost per layer
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    network.eval()
    exit_stats = ExitStatistics(num_exits=len(network.exit_layers))

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            # Get predictions at specific exit
            _, exit_data = network(features, return_all_exits=True)
            logits = exit_data[f"logits_{exit_layer}"]
            predictions = logits.argmax(dim=-1)

            # Compute cost
            cost = sum(cost_per_layer[: exit_layer + 1])

            # Update statistics
            for i in range(len(labels)):
                is_correct = bool(predictions[i] == labels[i])
                pos_idx = network.exit_layers.index(exit_layer)
                exit_stats.update(pos_idx, is_correct, cost)

    return exit_stats.get_metrics()


def evaluate_oracle_random(
    network: nn.Module,
    test_loader,
    cost_per_layer: list,
    device: str,
    seed: int = 42,
) -> dict:
    """
    Oracle baseline: randomly select exit for each sample.
    """
    network.eval()
    exit_stats = ExitStatistics(num_exits=len(network.exit_layers))
    rng = np.random.RandomState(seed)

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            _, exit_data = network(features, return_all_exits=True)

            for i in range(len(labels)):
                # Randomly select exit layer id
                exit_layer = int(rng.choice(network.exit_layers))

                logits = exit_data[f"logits_{exit_layer}"][i : i + 1]
                prediction = logits.argmax(dim=-1)
                is_correct = bool(prediction == labels[i])

                cost = sum(cost_per_layer[: exit_layer + 1])
                pos_idx = network.exit_layers.index(exit_layer)
                exit_stats.update(pos_idx, is_correct, cost)

    return exit_stats.get_metrics()


def evaluate_oracle_confidence(
    network: nn.Module,
    test_loader,
    threshold: float,
    cost_per_layer: list,
    device: str,
) -> dict:
    """
    Oracle baseline: exit early if confidence > threshold.
    """
    network.eval()
    exit_stats = ExitStatistics(num_exits=len(network.exit_layers))

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            _, exit_data = network(features, return_all_exits=True)

            for i in range(len(labels)):
                exited = False

                # Try each exit in order
                for exit_layer in network.exit_layers[:-1]:
                    logits = exit_data[f"logits_{exit_layer}"][i : i + 1]
                    probs = torch.softmax(logits, dim=-1)
                    confidence = probs.max().item()

                    if confidence > threshold:
                        prediction = logits.argmax(dim=-1)
                        is_correct = bool(prediction == labels[i])
                        cost = sum(cost_per_layer[: exit_layer + 1])

                        pos_idx = network.exit_layers.index(exit_layer)
                        exit_stats.update(pos_idx, is_correct, cost)
                        exited = True
                        break

                if not exited:
                    # Use final exit
                    final_layer = network.exit_layers[-1]
                    logits = exit_data[f"logits_{final_layer}"][i : i + 1]
                    prediction = logits.argmax(dim=-1)
                    is_correct = bool(prediction == labels[i])
                    cost = sum(cost_per_layer[: final_layer + 1])

                    pos_idx = network.exit_layers.index(final_layer)
                    exit_stats.update(pos_idx, is_correct, cost)

    return exit_stats.get_metrics()


def evaluate_learned_routing(
    network: nn.Module,
    routing_module: nn.Module,
    test_loader,
    cost_per_layer: list,
    device: str,
) -> dict:
    """
    Evaluate learned RL-based routing policy.
    """
    network.eval()
    routing_module.eval()
    exit_stats = ExitStatistics(num_exits=len(network.exit_layers))

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            _, exit_data = network(features, return_all_exits=True)

            for i in range(len(labels)):
                exited = False

                # Try each exit with routing module
                for exit_layer in network.exit_layers[:-1]:
                    features_at_exit = exit_data[f"features_{exit_layer}"][i : i + 1]
                    exit_prob = routing_module(features_at_exit, exit_layer, context=None)

                    if exit_prob > 0.5:
                        logits = exit_data[f"logits_{exit_layer}"][i : i + 1]
                        prediction = logits.argmax(dim=-1)
                        is_correct = bool(prediction == labels[i])
                        cost = sum(cost_per_layer[: exit_layer + 1])

                        pos_idx = network.exit_layers.index(exit_layer)
                        exit_stats.update(pos_idx, is_correct, cost)
                        exited = True
                        break

                if not exited:
                    # Use final exit
                    final_layer = network.exit_layers[-1]
                    logits = exit_data[f"logits_{final_layer}"][i : i + 1]
                    prediction = logits.argmax(dim=-1)
                    is_correct = bool(prediction == labels[i])
                    cost = sum(cost_per_layer[: final_layer + 1])

                    pos_idx = network.exit_layers.index(final_layer)
                    exit_stats.update(pos_idx, is_correct, cost)

    return exit_stats.get_metrics()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MIDAS early-exit network")
    parser.add_argument(
        "--checkpoint", type=str, default="best_model.pt", help="Path to model checkpoint"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save CSV/JSON results")
    args = parser.parse_args()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network, routing_module, config = load_model(args.checkpoint, device)

    # Override config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Load test data
    if "train_path" in config["data"]:
        _, _, test_loader, _ = load_split_netflow_data(
            train_path=config["data"]["train_path"],
            val_path=config["data"]["val_path"],
            test_path=config["data"]["test_path"],
            target_column=config["data"]["target_column"],
            normalize=config["data"]["normalize"],
            batch_size=config["data"]["batch_size"],
        )
    else:
        _, _, test_loader, _ = load_netflow_data(
            data_path=config["data"]["data_path"],
            target_column=config["data"]["target_column"],
            test_size=config["data"]["test_size"],
            val_size=config["data"]["val_size"],
            normalize=config["data"]["normalize"],
            random_state=config["data"]["random_seed"],
        )

    cost_per_layer = config["cost"]["cost_per_layer"]
    output_dir = args.output_dir or config.get("evaluation", {}).get("output_dir", "results")

    # Initialize W&B for logging results
    if config["wandb"]["enabled"]:
        wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"]["entity"],
            name=f"eval_{config['wandb'].get('run_name', 'model')}",
            tags=config["wandb"]["tags"] + ["evaluation"],
            config=config,
        )

    results = {}
    max_cost = sum(cost_per_layer)

    # Evaluate learned routing
    if routing_module is not None:
        print("\nEvaluating learned routing policy...")
        metrics = evaluate_learned_routing(network, routing_module, test_loader, cost_per_layer, device)
        results["learned_routing"] = metrics
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Avg Cost: {metrics['avg_cost']:.4f} (norm={metrics['avg_cost']/max_cost:.4f})")
        print(f"  Accuracy/Cost: {metrics['accuracy_per_cost']:.4f}")

    # Evaluate oracle baselines
    print("\n" + "=" * 50)
    print("Oracle Baselines:")
    print("=" * 50)

    for exit_layer in network.exit_layers:
        print(f"\nAlways Exit {exit_layer}...")
        metrics = evaluate_oracle_always_exit(network, test_loader, exit_layer, cost_per_layer, device)
        results[f"oracle_exit_{exit_layer}"] = metrics
        print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  Avg Cost: {metrics['avg_cost']:.4f} (norm={metrics['avg_cost']/max_cost:.4f})")
        print(f"  Accuracy/Cost: {metrics['accuracy_per_cost']:.4f}")

    # Random baseline
    print("\nRandom Exit Selection...")
    metrics = evaluate_oracle_random(network, test_loader, cost_per_layer, device)
    results["oracle_random"] = metrics
    print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Avg Cost: {metrics['avg_cost']:.4f} (norm={metrics['avg_cost']/max_cost:.4f})")
    print(f"  Accuracy/Cost: {metrics['accuracy_per_cost']:.4f}")

    # Confidence-based baseline
    threshold = config["evaluation"].get("confidence_threshold", 0.9)
    print(f"\nConfidence-based (threshold={threshold})...")
    metrics = evaluate_oracle_confidence(
        network, test_loader, threshold, cost_per_layer, device
    )
    results["oracle_confidence"] = metrics
    print(f"  Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  Avg Cost: {metrics['avg_cost']:.4f} (norm={metrics['avg_cost']/max_cost:.4f})")
    print(f"  Accuracy/Cost: {metrics['accuracy_per_cost']:.4f}")

    # Compute normalized cost and persist results
    for method, m in results.items():
        if "avg_cost" in m and max_cost > 0:
            m["avg_cost_normalized"] = m["avg_cost"] / max_cost

    # Save CSV/JSON
    from pathlib import Path
    import json, csv, time
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"evaluation_{stamp}.csv"
    json_path = out_dir / f"evaluation_{stamp}.json"

    fieldnames = ["method", "overall_accuracy", "avg_cost", "avg_cost_normalized", "accuracy_per_cost"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, m in results.items():
            writer.writerow({
                "method": method,
                "overall_accuracy": m.get("overall_accuracy"),
                "avg_cost": m.get("avg_cost"),
                "avg_cost_normalized": m.get("avg_cost_normalized"),
                "accuracy_per_cost": m.get("accuracy_per_cost"),
            })

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {csv_path} and {json_path}")

    # Log all results to W&B
    if config["wandb"]["enabled"]:
        # Flatten results for logging
        for method, metrics in results.items():
            for metric_name, value in metrics.items():
                wandb.log({f"{method}/{metric_name}": value})

        # Optionally upload outputs as artifacts
        try:
            art = wandb.Artifact("evaluation_results", type="results")
            art.add_file(str(csv_path))
            art.add_file(str(json_path))
            wandb.log_artifact(art)
        except Exception:
            pass

        wandb.finish()

    # Print summary comparison
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"{'Method':<25} {'Accuracy':<12} {'Cost':<12} {'Acc/Cost':<12} {'NormCost':<10}")
    print("-" * 50)

    for method, metrics in results.items():
        acc = metrics['overall_accuracy']
        cost = metrics['avg_cost']
        acc_per_cost = metrics['accuracy_per_cost']
        print(f"{method:<25} {acc:<12.4f} {cost:<12.4f} {acc_per_cost:<12.4f} {cost/max_cost:<10.4f}")

    # If learned routing present, show savings vs final-exit baseline
    if 'learned_routing' in results:
        final_key = f"oracle_exit_{network.exit_layers[-1]}"
        if final_key in results:
            lr_cost = results['learned_routing']['avg_cost']
            final_cost = results[final_key]['avg_cost']
            abs_saving = final_cost - lr_cost
            rel_saving = abs_saving / final_cost if final_cost > 0 else 0.0
            print("\nSavings vs Always Final:")
            print(f"  Absolute: {abs_saving:.4f} cost units")
            print(f"  Relative: {rel_saving*100:.2f}%")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
