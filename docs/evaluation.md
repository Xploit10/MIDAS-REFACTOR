Evaluation
==========

Command
- `python evaluate.py --config config_routing.yaml --checkpoint checkpoints/baseline_best.pt`

Outputs
- Learned routing metrics: accuracy, average cost, accuracy per cost.
- Oracle baselines:
  - Always exit at each configured exit layer
  - Random exit selection
  - Confidence-based exit (threshold from `evaluation.confidence_threshold`)
- Normalized cost is shown as `avg_cost / sum(cost_per_layer)`.
- Summary table lists all methods and a savings report vs the final exit baseline.

Sanity Checks
- Verify “Always Final” has the highest cost and near-best accuracy.
- Learned routing should reduce cost relative to “Always Final” with small accuracy drop.

Tips
- If normalized cost looks too high, increase `cost.lambda` to encourage earlier exits.
- Consider adding a FLOPs-based cost model for hardware-agnostic analysis (see docs/costs.md).

