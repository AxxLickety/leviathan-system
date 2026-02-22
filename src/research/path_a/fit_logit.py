from __future__ import annotations

import pandas as pd
import statsmodels.api as sm


def fit_interaction_logit(df: pd.DataFrame):
    df = df.copy()
    df["dti_x_regime"] = df["dti"] * df["regime"]

    X = sm.add_constant(df[["dti", "regime", "dti_x_regime"]])
    y = df["y"]

    model = sm.Logit(y, X)
    res = model.fit(disp=False)

    df["pred_prob"] = res.predict(X)
    return res, df
