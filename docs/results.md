Results and Verification
========================

During Training
- Watch `Val Accuracy` and `Val Avg Cost (norm)` per epoch.
- Expect cost to decrease (or stay low) while accuracy remains near the final-exit baseline.

After Training
- Run evaluation to get a comparison table across methods (learned routing and baselines).
- Key indicators:
  - Lower average cost than "Always Final".
  - Minimal accuracy drop compared to "Always Final".
  - Better `accuracy_per_cost` than naive baselines.

Savings vs Final
- The evaluator prints absolute and relative cost savings vs the final exit baseline.
- This quantifies the true benefit of early exit under your cost model.

Common Pitfalls
- If savings are small: reduce `threshold` in confidence baseline for context, or raise `cost.lambda` to push earlier exits.
- If accuracy drops too much: lower `cost.lambda`, or improve exits (e.g., add dropout, adjust exit layers).

