def apply_regime_filter(
    df,
    signal_col="score_xs",
    active_col="affordability_active",
):
    df = df.copy()
    df["signal_naive"] = df[signal_col]
    df["signal_filtered"] = df[signal_col] * df[active_col]
    return df

