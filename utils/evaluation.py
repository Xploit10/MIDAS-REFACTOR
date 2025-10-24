import numpy as np

def aggregate_results(evaluation_df):
    N = len(evaluation_df)
    metrics = {}

    # System confusion matrix (final prediction for each sample)
    system_tp = evaluation_df["tp"].sum() + evaluation_df["l_two_tp"].sum()
    system_tn = evaluation_df["tn"].sum() + evaluation_df["l_two_tn"].sum()
    system_fp = evaluation_df["fp"].sum() + evaluation_df["l_two_fp"].sum()
    system_fn = evaluation_df["fn"].sum() + evaluation_df["l_two_fn"].sum()
    metrics["system_tp"] = system_tp
    metrics["system_tn"] = system_tn
    metrics["system_fp"] = system_fp
    metrics["system_fn"] = system_fn
    metrics["system_acc"] = (system_tp + system_tn) / N

    # L1 metrics (where L2 was NOT used)
    l1_mask = evaluation_df["l_two_count"] == 0
    l1_N = l1_mask.sum()
    metrics["l1_tp"] = evaluation_df.loc[l1_mask, "tp"].sum()
    metrics["l1_tn"] = evaluation_df.loc[l1_mask, "tn"].sum()
    metrics["l1_fp"] = evaluation_df.loc[l1_mask, "fp"].sum()
    metrics["l1_fn"] = evaluation_df.loc[l1_mask, "fn"].sum()
    metrics["l1_acc"] = (metrics["l1_tp"] + metrics["l1_tn"]) / l1_N if l1_N else np.nan

    # L2 metrics (where L2 was used)
    l2_mask = evaluation_df["l_two_count"] == 1
    l2_N = l2_mask.sum()
    metrics["l2_tp"] = evaluation_df.loc[l2_mask, "l_two_tp"].sum()
    metrics["l2_tn"] = evaluation_df.loc[l2_mask, "l_two_tn"].sum()
    metrics["l2_fp"] = evaluation_df.loc[l2_mask, "l_two_fp"].sum()
    metrics["l2_fn"] = evaluation_df.loc[l2_mask, "l_two_fn"].sum()
    metrics["l2_acc"] = (metrics["l2_tp"] + metrics["l2_tn"]) / l2_N if l2_N else np.nan

    # L2 usage rate (cost-saving metric)
    metrics["layer2_usage_rate"] = evaluation_df["l_two_count"].mean()
    metrics["l2_calls"] = int(evaluation_df["l_two_count"].sum())
    metrics["l1_calls"] = int(N - metrics["l2_calls"])

    # Cost-saving estimate: set C1, C2 as your real compute costs
    C1 = 1   # cost per L1 call
    C2 = 10  # cost per L2 call (incremental, i.e., L1+L2 is 1+10=11)
    metrics["total_cost"] = metrics["l1_calls"] * C1 + metrics["l2_calls"] * (C1 + C2)
    # Normalize: cost vs. always using L2 for all
    metrics["relative_cost"] = metrics["total_cost"] / (N * (C1 + C2))

    # Reward/regret averages
    metrics["avg_reward"] = evaluation_df["reward"].mean()
    metrics["avg_regret_benign"] = evaluation_df["accumulated_regret_benign"].mean()
    metrics["avg_regret_phishing"] = evaluation_df["accumulated_regret_phishing"].mean()

    ALPHA = 0.5  # or sweep this to see tradeoffs!

    metrics["effective_accuracy"] = metrics["system_acc"] - ALPHA * metrics["relative_cost"]
    metrics["acc_per_cost"] = metrics["system_acc"] / metrics["relative_cost"] if metrics["relative_cost"] > 0 else np.nan

    if "gate_probability" in evaluation_df:
        metrics["avg_gate_probability"] = float(evaluation_df["gate_probability"].mean())
        metrics["gate_probability_std"] = float(evaluation_df["gate_probability"].std())
    if "dual_lambda" in evaluation_df:
        metrics["avg_dual_lambda"] = float(evaluation_df["dual_lambda"].mean())

    return metrics
