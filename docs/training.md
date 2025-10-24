Training Guide
==============

Objective
- Optimize the routing policy while keeping the backbone fixed.

Command
- `python train.py --config config_routing.yaml`

What Happens
- Loads frozen backbone from `training.load_checkpoint` if provided.
- Builds router (`routing.type`: `mlp` or `attention`).
- Prints a per-layer cost profile from `cost.cost_per_layer`.
- For each batch, collects exit logits and features at all exit layers.
- Computes a weighted classification loss (monitoring) and a routing RL loss.
- Applies policy gradient updates to the router only.
- Runs validation each epoch and shows:
  - `Val Accuracy`: accuracy under the current routing policy (threshold 0.5).
  - `Val Avg Cost`: average cumulative cost across exits.
  - `Val Avg Cost (norm)`: `Val Avg Cost / sum(cost_per_layer)`.

Key Configs
- `training.freeze_backbone: true` — ensure only the router trains.
- `training.lr_routing` — routing optimizer LR; `lr_classifier` can be 0 when frozen.
- `training.rl.lambda` via `cost.lambda` — increases weight of cost penalty.
- `training.exit_loss_weights` — weights for per-exit CE (monitoring).

Checkpoints
- Saved to `checkpoints/baseline_best.pt` on improved validation metric.
- Contains backbone, router, and optimizer states.

W&B
- Set `wandb.enabled: true` and fill `project` and `entity` to log metrics.

