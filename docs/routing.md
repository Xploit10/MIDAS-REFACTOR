Routing Policy
==============

Options
- MLP router: compact policy over features + layer embedding + optional context.
- Attention router: transformer blocks over two tokens (primary + interaction features) and layer embedding.

Training Objective
- REINFORCE with advantage baseline:
  - reward = accuracy − lambda × normalized_cost
  - advantage via EMA baseline (`training.rl.advantage_beta`).
  - entropy regularization (`training.rl.entropy_coef`) to encourage exploration.

Decisions
- During training: sample Bernoulli actions from the exit probability.
- During validation/evaluation: use a fixed threshold (default 0.5) for deterministic behavior.

Context Inputs
- Optional `context` vector (e.g., running stats) can be supplied if enabled in config.

Implementation
- See `models/routing_module.py`.

