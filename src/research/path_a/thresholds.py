from __future__ import annotations

import numpy as np
import pandas as pd


def compute_dti_thresholds(params: pd.Series, probs=(0.10, 0.20)):
    b0 = params["const"]
    b1 = params["dti"]
    b2 = params["regime"]
    b3 = params["dti_x_regime"]

    rows = []
    for r in [0, 1]:
        for p in probs:
            logit = np.log(p / (1 - p))
            dti_star = (logit - b0 - b2 * r) / (b1 + b3 * r)
            rows.append({"regime": r, "prob": p, "dti_threshold": dti_star})

    return pd.DataFrame(rows)
