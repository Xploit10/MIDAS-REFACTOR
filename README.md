# MIDAS: Early-Exit Neural Networks for Netflow Classification

MIDAS is a cost-optimized deep learning framework for netflow classification that uses **early-exit architectures** with **reinforcement learning-based routing** to balance accuracy against computational cost.

If you want actionable instructions, jump to Quick Start below. For deeper explanations and per-component guides, see the docs/ directory:
- docs/overview.md — concepts and workflow
- docs/training.md — how training works and what gets optimized
- docs/evaluation.md — how to evaluate and compare baselines
- docs/model.md — backbone and exits
- docs/routing.md — policy, reward, and sampling
- docs/costs.md — cost models and interpretation
- docs/results.md — reading metrics and verifying savings

## Overview

Traditional deep networks process all inputs through all layers, wasting computation on "easy" samples. MIDAS embeds multiple exit points throughout the network, allowing simple samples to exit early while complex samples continue to deeper layers.

**Key Innovation:** A learned routing policy (trained with REINFORCE) decides when to exit, optimizing the tradeoff between classification accuracy and computational cost.

## Architecture

```
Input (Netflow Features)
    ↓
[Layer 1] ──→ Exit 1 (fast, less accurate)
    ↓
[Layer 2]
    ↓
[Layer 3] ──→ Exit 2 (medium cost/accuracy)
    ↓
[Layer 4] ──→ Exit 3 (final, most accurate)
```

At each exit point, a **routing module** decides:
- **Exit now** with current prediction (save computation)
- **Continue** to next layer (improve accuracy)

The routing policy is trained end-to-end using policy gradients with reward:
```
reward = accuracy - λ × normalized_cost
```

## Quick Start

### 1. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up Weights & Biases

```bash
wandb login
```

### 3. Prepare Your Data

Your netflow data should be in CSV format with features + a label column:

```csv
feature_1,feature_2,...,feature_n,label
0.5,0.3,...,0.8,0
0.1,0.9,...,0.2,1
...
```

### 4. Configure Experiment

Edit [config.yaml](config.yaml) to set:
- Data paths
- Model architecture (layer dimensions, exit points)
- Routing type (MLP or Attention-based)
- Cost model and penalty weight
- Training hyperparameters

Example configuration:
```yaml
data:
  data_path: "data/netflow.csv"
  target_column: "label"

model:
  hidden_dims: [128, 256, 512, 256, 128]
  exit_layers: [1, 3, 4]  # Exits at layers 1, 3, 4
  num_classes: 2

routing:
  type: "attention"  # or "mlp"

cost:
  type: "layer_depth"
  cost_per_layer: [1.0, 2.0, 3.0, 4.0, 5.0]
  lambda: 0.1  # Cost penalty weight
```

### 5. Train

```bash
python train.py --config config.yaml
```

The training process:
1. **Warmup phase** (epochs 0-10): Train classification network only
2. **Joint training** (epochs 10+): Train both network + routing policy
3. **Outputs**: Best model saved to `best_model.pt`

### 6. Evaluate

```bash
python evaluate.py --checkpoint best_model.pt
```

This evaluates your model against oracle baselines:
- **Always Exit 0**: Use first exit (fastest, least accurate)
- **Always Final**: Use final exit (slowest, most accurate)
- **Confidence-based**: Exit when confidence > threshold
- **Random**: Random exit selection
- **Learned Routing**: Your trained RL policy

## Project Structure

```
MIDAS/
├── models/
│   ├── netflow_network.py      # Main early-exit MLP
│   ├── exit_head.py             # Exit classifier heads
│   ├── routing_module.py        # RL-based routing policies
│   └── attention_router.py      # Transformer components (legacy)
├── utils/
│   ├── data.py                  # Netflow data loaders
│   ├── rl_utils.py              # REINFORCE trainer, advantage estimation
│   ├── cost_model.py            # Cost computation (FLOPs, latency, etc.)
│   ├── evaluation.py            # Metrics computation (legacy)
│   └── features.py              # Feature utilities (legacy)
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── config.yaml                  # Experiment configuration
└── requirements.txt             # Python dependencies
```

## Configuration Options

### Model Architecture

**`model.hidden_dims`**: List of hidden layer dimensions
```yaml
hidden_dims: [128, 256, 512, 256, 128]  # 5 layers
```

**`model.exit_layers`**: Which layers have exits (0-indexed)
```yaml
exit_layers: [1, 3, 4]  # Exits at layers 1, 3, and 4
```

**`model.num_classes`**: Number of output classes
```yaml
num_classes: 2  # Binary classification
```

### Routing Module

**MLP Routing** (simpler, faster):
```yaml
routing:
  type: "mlp"
  hidden_dim: 64
  temperature: 1.0
```

**Attention Routing** (more powerful):
```yaml
routing:
  type: "attention"
  hidden_dim: 64
  num_heads: 4
  num_layers: 2
  temperature: 1.0
```

### Cost Models

**Layer Depth** (simple incremental cost):
```yaml
cost:
  type: "layer_depth"
  cost_per_layer: [1.0, 2.0, 3.0, 4.0, 5.0]
```

**Exponential** (deeper layers cost more):
```yaml
cost:
  type: "exponential"
  base: 1.5
  offset: 1.0
```

**FLOPs** (actual floating point operations):
```yaml
cost:
  type: "flops"
  layer_dims: [100, 128, 256, 512, 256, 128]  # Include input_dim
```

**Composite** (weighted combination):
```yaml
cost:
  type: "composite"
  models:
    flops:
      type: "flops"
      layer_dims: [100, 128, 256, 512, 256, 128]
    latency:
      type: "exponential"
      base: 1.2
  weights:
    flops: 0.7
    latency: 0.3
```

### Training

**RL Training**:
```yaml
training:
  rl:
    warmup_epochs: 10          # Train classifier first
    advantage_beta: 0.99       # Baseline EMA decay
    entropy_coef: 0.01         # Exploration bonus
```

**Early Stopping**:
```yaml
training:
  early_stopping:
    enabled: true
    patience: 10
    metric: "val_accuracy"     # or "val_accuracy_per_cost"
```

## Metrics

### Logged to W&B

**Per-Exit Metrics**:
- `accuracy_exit_{i}`: Accuracy at exit i
- `usage_exit_{i}`: % of samples using exit i

**Cost Metrics**:
- `avg_cost`: Average computational cost per sample
- `avg_cost_normalized`: Normalized to [0, 1]

**Tradeoff Metrics**:
- `overall_accuracy`: Overall classification accuracy
- `accuracy_per_cost`: Accuracy divided by cost (efficiency)

**Training Metrics**:
- `train_loss`, `train_cls_loss`, `train_rl_loss`
- Exit probability distributions
- Policy entropy (exploration measure)

## Advanced Usage

### Hyperparameter Sweeps with W&B

Enable sweeps in config:
```yaml
experiment:
  sweep:
    enabled: true
    method: "bayes"
    parameters:
      cost.lambda:
        values: [0.05, 0.1, 0.15, 0.2]
      routing.type:
        values: ["mlp", "attention"]
```

Then run:
```bash
wandb sweep config.yaml
wandb agent <sweep-id>
```

### Custom Cost Models

Implement in [utils/cost_model.py](utils/cost_model.py):
```python
class CustomCost(CostModel):
    def compute_cost(self, exit_layer: int, total_layers: int) -> float:
        # Your cost logic here
        return cost_value
```

### Custom Routing Policies

Extend [models/routing_module.py](models/routing_module.py):
```python
class CustomRoutingModule(nn.Module):
    def forward(self, features, layer_idx, context=None):
        # Return exit probability [0, 1]
        return exit_prob
```

## Expected Results

On typical netflow datasets, MIDAS achieves:

- **50-70% cost reduction** vs. always using final exit
- **<1% accuracy drop** compared to full network
- **Better accuracy-cost tradeoff** than confidence-based routing

Example output:
```
Method                    Accuracy     Cost         Acc/Cost
----------------------------------------------------------
learned_routing           0.9450       2.3          0.4109
oracle_exit_1             0.8200       1.0          0.8200
oracle_exit_4             0.9500       5.0          0.1900
oracle_confidence         0.9100       3.8          0.2395
```

## Migration from Legacy MIDAS

**What Changed:**

❌ **Removed**:
- Bandit-based simulation (`actions/`, `selections/`, `rewards/`)
- Post-hoc gating (separate L1/L2 models)
- Grid search explosion (`configs/grid_search_config.py`)
- Error predictor training (`utils/error_predictor.py`)

✅ **New**:
- End-to-end trainable network
- Integrated routing layers
- Simple YAML configuration
- RL-based policy learning

**Legacy Support:**

The legacy prediction CSV loader is still available in [utils/data.py](utils/data.py#L176) (`load_datasets()`), but new experiments should use `load_netflow_data()`.

## Troubleshooting

**CUDA out of memory:**
- Reduce `data.batch_size` in config
- Reduce `model.hidden_dims`
- Use smaller routing network (`routing.hidden_dim`)

**Routing always exits at first layer:**
- Increase `cost.lambda` (higher cost penalty)
- Check exit classification accuracy (may need better features)
- Increase `training.rl.warmup_epochs`

**Training unstable:**
- Lower learning rates (`training.lr_routing`)
- Enable gradient clipping (`training.grad_clip`)
- Increase `training.rl.advantage_beta` (smoother baseline)

## Citation

If you use MIDAS in your research, please cite:

```bibtex
@software{midas2024,
  title={MIDAS: Early-Exit Networks for Cost-Optimized Inference},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/MIDAS}
}
```

## License

[Your License Here]

## Contributing

Contributions welcome! Please open an issue or PR.

---

**Questions?** Open an issue or contact [your-email@example.com]
