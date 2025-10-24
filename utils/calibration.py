# utils/calibration.py  ✔ fixed
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def calibrate_probs(df_l1: pd.DataFrame, out_path: str | None = None):
    """
    Platt-scale the raw L1 probabilities so that, e.g.,
    0.70 really means ~70 % chance of class 1.

    df_l1 must contain columns:
        prediction – raw prob from the NN
        y          – ground-truth label (0/1)

    Returns the calibrated DataFrame; optionally saves it.
    """
    X = df_l1["prediction"].values.reshape(-1, 1)
    y = df_l1["y"].values

    # Fit a simple logistic reg on the 1-D score
    base = LogisticRegression().fit(X, y)

    # Wrap it in CalibratedClassifierCV; estimator= (not base_estimator)
    platt = CalibratedClassifierCV(
        estimator=base,        # <- fixed keyword
        cv="prefit",
        method="sigmoid",
    ).fit(X, y)

    # Replace raw probs with calibrated ones
    df_l1 = df_l1.copy()
    df_l1["prediction"] = platt.predict_proba(X)[:, 1]

    if out_path:
        df_l1.to_csv(out_path, index=False)

    return df_l1
