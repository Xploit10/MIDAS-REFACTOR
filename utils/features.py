import math
import numpy as np

EPS = 1e-9


def gating_feature_vector(
    l1_raw: float,
    l1_calibrated: float,
    l2_raw: float,
    ctx_value: float,
    usage_rate: float,
) -> np.ndarray:
    margin = abs(l1_calibrated - 0.5)
    entropy = -(l1_raw * math.log(l1_raw + EPS) + (1 - l1_raw) * math.log(1 - l1_raw + EPS))
    delta = l2_raw - l1_raw
    features = np.array(
        [
            l1_raw,
            l1_calibrated,
            margin,
            entropy,
            l2_raw,
            delta,
            abs(delta),
            ctx_value,
            usage_rate,
            usage_rate ** 2,
            margin * abs(delta),
        ],
        dtype=np.float32,
    )
    return features
