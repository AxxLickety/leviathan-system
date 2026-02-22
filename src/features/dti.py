import pandas as pd

def compute_price_to_income(df: pd.DataFrame) -> pd.Series:
    required = {"price", "income"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return df["price"] / df["income"]


def compute_mortgage_burden(
    df: pd.DataFrame,
    *,
    term_years: int = 25,
    loan_to_value: float = 0.8,
) -> pd.Series:
    required = {"price", "income", "mortgage_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    price = df["price"].astype(float)
    income = df["income"].astype(float)
    r_annual = df["mortgage_rate"].astype(float)

    r_month = r_annual / 12.0
    n_months = term_years * 12
    loan = price * loan_to_value

    def monthly_payment(P, r, n):
        eps = 1e-8
        if abs(r) < eps:
            return P / n
        factor = (1 + r) ** n
        return P * r * factor / (factor - 1)

    payment = monthly_payment(loan, r_month, n_months)
    monthly_income = income / 12.0
    return payment / monthly_income


def attach_affordability_features(
    df: pd.DataFrame,
    *,
    term_years: int = 25,
    loan_to_value: float = 0.8,
) -> pd.DataFrame:
    df_out = df.copy()

    df_out["dti_simple"] = compute_price_to_income(df_out)

    if "mortgage_rate" in df_out.columns:
        df_out["dti_mortgage"] = compute_mortgage_burden(
            df_out,
            term_years=term_years,
            loan_to_value=loan_to_value,
        )

    return df_out
