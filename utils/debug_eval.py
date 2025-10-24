import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def debug_layer_performance(l1_df, l2_df, print_results=True):
    assert len(l1_df) == len(l2_df), "L1 and L2 datasets must have same length!"
    y_true = np.array(l1_df["y"])
    l1_probs = np.array(l1_df["prediction"])
    l2_probs = np.array(l2_df["prediction"])
    l1_pred = (l1_probs > 0.5).astype(int)
    l2_pred = (l2_probs > 0.5).astype(int)

    # Stand-alone L1
    l1_acc = np.mean(l1_pred == y_true)
    l1_cm = confusion_matrix(y_true, l1_pred)
    
    # Stand-alone L2
    l2_acc = np.mean(l2_pred == y_true)
    l2_cm = confusion_matrix(y_true, l2_pred)

    # How often does L2 correct L1?
    l1_wrong = l1_pred != y_true
    l2_corrects = np.sum(l1_wrong & (l2_pred == y_true))
    l1_wrong_total = np.sum(l1_wrong)
    l2_correction_rate = l2_corrects / l1_wrong_total if l1_wrong_total > 0 else 0

    # How often does L2 make it worse?
    l2_makes_worse = np.sum((l1_pred == y_true) & (l2_pred != y_true))
    l1_right_total = np.sum(l1_pred == y_true)
    l2_degradation_rate = l2_makes_worse / l1_right_total if l1_right_total > 0 else 0

    # Always L1 system acc (baseline)
    always_l1_acc = l1_acc

    # Always L2 system acc (oracle)
    always_l2_acc = l2_acc

    if print_results:
        print("Stand-alone L1 accuracy:", l1_acc)
        print("Stand-alone L1 confusion matrix:\n", l1_cm)
        print("\nStand-alone L2 accuracy:", l2_acc)
        print("Stand-alone L2 confusion matrix:\n", l2_cm)
        print("\nL2 corrects L1 errors:", l2_corrects, f"out of {l1_wrong_total} L1 mistakes ({100*l2_correction_rate:.2f}%)")
        print("L2 makes correct L1 wrong:", l2_makes_worse, f"out of {l1_right_total} correct L1 predictions ({100*l2_degradation_rate:.2f}%)")
        print("\nAlways-L1 system accuracy:", always_l1_acc)
        print("Always-L2 system accuracy (oracle):", always_l2_acc)

    return {
        "l1_acc": l1_acc,
        "l2_acc": l2_acc,
        "l1_cm": l1_cm,
        "l2_cm": l2_cm,
        "l2_correction_rate": l2_correction_rate,
        "l2_degradation_rate": l2_degradation_rate,
        "l2_corrects": l2_corrects,
        "l2_makes_worse": l2_makes_worse
    }

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file paths
    l1 = pd.read_csv("./prediction_files/dataset_bell_nn_l1.csv")
    l2 = pd.read_csv("./prediction_files/dataset_bell_nn_l2.csv")
    debug_layer_performance(l1, l2)
