MIDAS Overview
===============

Goal
- Learn a routing policy that exits early for easy inputs and continues deeper for hard ones to reduce inference cost with minimal accuracy loss.

Workflow
- Train routing with a frozen backbone using policy gradients (REINFORCE).
- Validate per epoch to track accuracy and average cost.
- Evaluate on the test set with baselines to quantify savings.

Key Pieces
- Backbone: multi-exit MLP (`models/netflow_network.py` + `models/exit_head.py`).
- Router: MLP or attention-based policy (`models/routing_module.py`).
- Cost model: defines per-layer cost (`utils/cost_model.py`).
- RL utilities: advantage estimator + REINFORCE (`utils/rl_utils.py`).

Config Sources
- `config_routing.yaml`: frozen backbone, routing-only training.
- `config.yaml` / `config_baseline.yaml`: alternative experiments.

Run Paths
- Train routing: `python train.py --config config_routing.yaml`.
- Evaluate: `python evaluate.py --config config_routing.yaml --checkpoint checkpoints/baseline_best.pt`.

