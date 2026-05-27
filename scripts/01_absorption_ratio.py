"""
Step 01 — Absorption ratio, risk decomposition, and benchmark alignment.

Outputs (to `outputs/`):
    step01_descriptive_stats_funds.csv
    step01_descriptive_stats_global.csv
    step01_ar_baseline.csv
    step01_ar_augmented.csv
    step01_ar_correlations.csv
    step01_risk_decomposition.csv
    step01_regularisation_verification.csv

Figures (to `figures/`):
    fig01_return_series.pdf
    fig02_ar_benchmarks_events.pdf
    fig04_ar_baseline_vs_augmented.pdf
    fig11_risk_decomposition.pdf   (regime overlay added in Step 03)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irdpfn.absorption import (
    compute_absorption_ratio,
    compute_risk_decomposition,
    tracking_error,
    verify_regularisation,
)
from irdpfn.config import N_FUNDS, OUTPUTS_DIR, ROLLING_WINDOW
from irdpfn.data_io import descriptive_stats, load_and_align
from irdpfn.figures import (
    fig01_return_series,
    fig02_ar_benchmarks_events,
    fig04_ar_baseline_vs_augmented,
)


def main():
    # 1. Load and align everything
    df, R_f, R_bf, R_bg, R_aug = load_and_align()
    print(f"\nLoaded panel: {df.shape}")
    print(f"R_f   shape: {R_f.shape}")
    print(f"R_bf  shape: {R_bf.shape}")
    print(f"R_bg  shape: {R_bg.shape}")
    print(f"R_aug shape: {R_aug.shape}")
    print(f"Date range: {R_f.index.min().date()} to {R_f.index.max().date()}")

    # 2. Descriptive statistics
    desc_funds  = descriptive_stats(df.set_index("Date")["log_return_price"]
                                      .rename("Full sample"))
    desc_global = descriptive_stats(R_bg)
    desc_funds.to_csv(OUTPUTS_DIR / "step01_descriptive_stats_funds.csv")
    desc_global.to_csv(OUTPUTS_DIR / "step01_descriptive_stats_global.csv")

    # 3. Baseline AR_t (from R_f only)
    print(f"\nComputing baseline AR_t (window = {ROLLING_WINDOW})...")
    AR_baseline = compute_absorption_ratio(R_f, window=ROLLING_WINDOW).dropna()
    AR_baseline.to_csv(OUTPUTS_DIR / "step01_ar_baseline.csv")
    print(f"  range: [{AR_baseline.min():.4f}, {AR_baseline.max():.4f}]")
    print(f"  mean:  {AR_baseline.mean():.4f}  std: {AR_baseline.std():.4f}")

    # 4. Augmented AR_t (robustness)
    print(f"\nComputing augmented AR_t...")
    AR_augmented = compute_absorption_ratio(R_aug, window=ROLLING_WINDOW).dropna()
    AR_augmented.to_csv(OUTPUTS_DIR / "step01_ar_augmented.csv")
    common = AR_baseline.index.intersection(AR_augmented.index)
    rho = AR_baseline.loc[common].corr(AR_augmented.loc[common])
    print(f"  correlation (baseline vs augmented): {rho:.4f}")

    # 5. AR correlations with benchmarks
    te = tracking_error(R_f, R_bf, kind="mae")
    R_bg_aligned = R_bg.reindex(AR_baseline.index)
    interp = pd.concat([AR_baseline,
                        te.reindex(AR_baseline.index),
                        R_bg_aligned], axis=1)
    interp.columns = ["AR", "Tracking_Error"] + list(R_bg.columns)
    corrs = interp.corr()["AR"].drop("AR")
    corrs.to_csv(OUTPUTS_DIR / "step01_ar_correlations.csv",
                 header=["correlation"])
    print(f"\nAR correlations with benchmarks:")
    print(corrs.round(4))

    # 6. Risk decomposition
    risk_decomp = compute_risk_decomposition(R_f, window=ROLLING_WINDOW)
    risk_decomp.to_csv(OUTPUTS_DIR / "step01_risk_decomposition.csv")

    # 7. Regularisation verification (footnote)
    verif = verify_regularisation(R_f, window=ROLLING_WINDOW)
    verif.to_csv(OUTPUTS_DIR / "step01_regularisation_verification.csv",
                 index=False)
    print(f"\nRegularisation verification:")
    print(verif.to_string(index=False))

    # 8. Figures (the ones that don't need HMM regimes)
    print(f"\nGenerating figures...")
    fig01_return_series(R_f, R_bf, R_bg, AR_baseline.index)
    fig02_ar_benchmarks_events(AR_baseline, R_bf, R_bg)
    fig04_ar_baseline_vs_augmented(AR_baseline, AR_augmented)

    print("\nStep 01 complete.")


if __name__ == "__main__":
    main()
