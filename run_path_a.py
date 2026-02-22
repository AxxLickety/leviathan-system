from pathlib import Path
import pandas as pd

from src.research.path_a.build_dataset import build_master_df
from src.research.path_a.label_correction import add_correction_label
from src.research.path_a.fit_logit import fit_interaction_logit
from src.research.path_a.thresholds import compute_dti_thresholds
from src.research.path_a.report import excerpt_first_last, top_risk_quarters


def main():
    print("[PATH_A] START")

    out = Path("outputs/path_a")
    out.mkdir(parents=True, exist_ok=True)

    master = build_master_df()
    master.to_csv(out / "master.csv", index=False)

    labeled = add_correction_label(master)
    labeled.to_csv(out / "labeled.csv", index=False)

    res, df_pred = fit_interaction_logit(labeled)
    df_pred.to_csv(out / "predictions.csv", index=False)

    coef = pd.DataFrame({
        "coef": res.params,
        "p_value": res.pvalues,
    })
    coef.to_csv(out / "coef.csv")

    thresholds = compute_dti_thresholds(res.params)
    thresholds.to_csv(out / "thresholds.csv", index=False)

    excerpt_first_last(df_pred).to_csv(out / "excerpt.csv", index=False)
    top_risk_quarters(df_pred).to_csv(out / "top5.csv", index=False)

    print("[PATH_A] DONE")


if __name__ == "__main__":
    main()
