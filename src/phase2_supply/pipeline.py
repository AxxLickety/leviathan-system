from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _load_csv(root: Path, rel: str) -> pd.DataFrame:
    p = root / rel
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p, parse_dates=["date"]).sort_values("date")


def _load_supply_optional(root: Path, rel: str) -> Optional[pd.DataFrame]:
    p = root / rel
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    return df


def _pick_supply_metric(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            # must have at least some non-null
            if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
                return c
    return None


def run(cfg: dict, root: Path) -> None:
    master = _load_csv(root, cfg["master_csv"])
    supply = _load_supply_optional(root, cfg["supply_csv"])

    if supply is None:
        df = master.copy()
    else:
        df = master.merge(supply, on="date", how="left")

    # Ensure forward 4Q return exists (needed for any Phase 2 evaluation)
    if "fwd_ret_4q" not in df.columns:
        if "real_price_index" not in df.columns:
            raise ValueError("master.csv must contain real_price_index to compute fwd_ret_4q")
        import numpy as np

        df["log_price"] = np.log(df["real_price_index"].astype(float))
        df["fwd_ret_4q"] = df["log_price"].shift(-4) - df["log_price"]
        df.drop(columns=["log_price"], inplace=True)

    metric = _pick_supply_metric(df, cfg["supply_candidates"])
    if metric is None:
        df["supply_metric_name"] = "NONE"
        df["supply_high"] = 0
    else:
        s_raw = pd.to_numeric(df[metric], errors="coerce")
        df["supply_value"] = s_raw

        s = s_raw

        method = str(cfg.get("gate", {}).get("method", "percentile"))
        thr = float(cfg["gate"]["threshold"])

        if method == "rolling_percentile":
            win = int(cfg["gate"].get("window", 40))
            lag = int(cfg["gate"].get("lag", 1))

            q = pd.Series(index=df.index, dtype=float)
            for r in df["regime"].dropna().unique():
                mask = df["regime"] == r
                s_r = s[mask]
                q_r = s_r.rolling(win, min_periods=max(8, win//4)).quantile(thr).shift(lag)
                q.loc[mask] = q_r
        else:
            q = s.quantile(thr)

        df["supply_metric_name"] = metric
        df["supply_high"] = (s >= q).astype("Int64").fillna(0).astype(int)
        df["supply_q"] = q

    out = root / cfg["outputs"]["sanity_csv"]
    out.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        c
        for c in [
            "date",
            "dti",
            "real_rate",
            "regime",
            "real_price_index",
            "fwd_ret_4q",
            "supply_metric_name",
            "supply_high","supply_q","permits","supply_value",
        ]
        if c in df.columns
    ]
    df[cols].to_csv(out, index=False)

    print(f"[phase2] wrote: {out}")
    print(df[cols].tail(10).to_string(index=False))
