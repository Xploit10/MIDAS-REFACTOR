Cost Models and Interpretation
==============================

Configuration
- In `config_routing.yaml: cost` choose a cost model and parameters.

Layer Depth (default)
- `cost_per_layer` defines incremental cost added by each layer.
- The code prints per-layer and cumulative cost at startup for transparency.

FLOPs
- Use `type: flops` with `layer_dims` (include `input_dim`).
- Approximates FC layer FLOPs as `2 * in_dim * out_dim`.

Latency
- Profiles average per-layer latency on a given device.
- Useful when hardware differences matter; profiling is done with dummy inputs.

Composite
- Blend multiple cost models with weights.

Normalized Cost
- We also report `avg_cost / sum(cost_per_layer)` for easy comparison across configs.

Choosing Lambda
- `cost.lambda` controls how aggressively the router saves cost.
- Higher lambda → earlier exits, lower cost, potential accuracy drop.

