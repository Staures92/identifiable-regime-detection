"""
Step 04 — Cross-step diagnostics.

This step is run after Steps 01-03 have written their outputs.

- Correlation significance (Pearson + Spearman + Kendall) for AR_t
  against tracking error and global benchmarks.
- Regime-conditional OLS regression with Newey-West HAC standard errors.
- Cluster x regime tracking error amplification.
- Covariance estimator comparison (sample / Ledoit-Wolf / MCD).

Outputs:
    step04_correlation_significance.csv
    step04_regime_conditional_regression.csv
    step04_cluster_regime_tracking_error.csv
    step04_covariance_comparison.csv

Figures:
    fig08_cluster_regime_tracking_error.pdf
    fig09_scatter_correlations.pdf
    fig10_regime_conditional_scatter.pdf
    fig12_covariance_comparison.pdf
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irdpfn.absorption import (
    compute_absorption_ratio,
    compute_ar_with_estimator,
    tracking_error,
)
from irdpfn.config import K_CLUSTER_STAR, OUTPUTS_DIR, REGIME_NAMES
from irdpfn.clustering import cluster_composition, compute_dtw_linkage
from irdpfn.data_io import load_and_align
from irdpfn.diagnostics import (
    build_interp_df,
    cluster_regime_tracking_error,
    correlation_tests,
    regime_conditional_regression,
)
from irdpfn.figures import (
    fig08_cluster_regime_tracking_error,
    fig09_scatter_correlations,
    fig10_regime_conditional_scatter,
    fig12_covariance_comparison,
)


def main():
    # 1. Reload data and recompute AR_t
    df, R_f, R_bf, R_bg, _ = load_and_align()
    AR_baseline = compute_absorption_ratio(R_f).dropna()

    # 2. Load regime labels from Step 03
    regime_path = OUTPUTS_DIR / "step03_regime_labels.csv"
    if not regime_path.exists():
        raise FileNotFoundError(
            "Run scripts/03_regime_detection.py first — "
            f"missing {regime_path}",
        )
    regime_series = pd.read_csv(regime_path, index_col=0,
                            parse_dates=True)["Regime"]

    # 3. Crisis threshold
    thresholds_df = pd.read_csv(OUTPUTS_DIR / "step03_thresholds.csv")
    tau = float(thresholds_df["tau"].iloc[0])

    # 4. Build the unified interp dataframe
    te = tracking_error(R_f, R_bf, kind="mae")
    interp = build_interp_df(AR_baseline, te, R_bg, regime_series)

    series_cols = ["Tracking_Error"] + list(R_bg.columns)

    # 5. Correlation tests
    print("\n" + "=" * 60)
    print("Correlation tests (Pearson / Spearman / Kendall)")
    print("=" * 60)
    corr_table = correlation_tests(interp, target_col="AR",
                                   series_cols=series_cols)
    corr_table.to_csv(
        OUTPUTS_DIR / "step04_correlation_significance.csv", index=False,
    )
    print(corr_table.to_string(index=False))

    # 6. Regime-conditional regression
    print("\n" + "=" * 60)
    print("Regime-conditional OLS regression (HAC SE)")
    print("=" * 60)
    fitted, reg_table = regime_conditional_regression(
        interp, series_cols=series_cols,
    )
    reg_table.to_csv(
        OUTPUTS_DIR / "step04_regime_conditional_regression.csv", index=False,
    )
    print(reg_table.to_string(index=False))

    # 7. Cluster x regime tracking error
    print("\n" + "=" * 60)
    print("Cluster x regime tracking error amplification")
    print("=" * 60)
    Z, R_f_scaled, fund_names = compute_dtw_linkage(R_f)
    cluster_df, _ = cluster_composition(Z, fund_names, R_f, R_bf, K_CLUSTER_STAR)

    cluster_te = cluster_regime_tracking_error(
        cluster_df, R_f, R_bf, AR_baseline, regime_series,
    )
    cluster_te.drop(columns=["Funds"]).to_csv(
        OUTPUTS_DIR / "step04_cluster_regime_tracking_error.csv",
    )
    print(cluster_te[["N_funds", "TE_normal", "TE_crisis", "Ratio",
                      "Cohort_type"]].round(6).to_string())

    # 8. Covariance estimator comparison
    print("\n" + "=" * 60)
    print("Covariance estimator comparison")
    print("=" * 60)
    ar_lw  = compute_ar_with_estimator(R_f, estimator="ledoit_wolf")
    ar_mcd = compute_ar_with_estimator(R_f, estimator="mcd")
    common = AR_baseline.index.intersection(ar_lw.index).intersection(ar_mcd.index)
    cov_compare = pd.DataFrame({
        "Sample":      AR_baseline.loc[common],
        "Ledoit_Wolf": ar_lw.loc[common],
        "MCD":         ar_mcd.loc[common],
    })
    cov_compare.to_csv(OUTPUTS_DIR / "step04_covariance_comparison.csv")
    print(cov_compare.describe().round(4))
    print(f"\nPairwise correlations:")
    print(cov_compare.corr().round(4))

    # 9. Figures
    fig08_cluster_regime_tracking_error(cluster_te)
    fig09_scatter_correlations(interp, tau=tau)
    fig10_regime_conditional_scatter(interp, fitted, tau=tau)
    fig12_covariance_comparison(cov_compare)

    print("\nStep 04 complete.")


if __name__ == "__main__":
    main()
