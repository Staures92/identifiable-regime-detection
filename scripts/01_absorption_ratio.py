"""
Step 01 — Absorption ratio and its robustness variants.

Computes the baseline covariance AR_t on the fund panel R^(f), the scale-free
correlation AR_t, and the augmented correlation AR_t on
A = [R^(f) | R^(b,f) | G^(b)] (N = 85). It then runs the supporting checks the
referees asked for: the materiality of ridge regularisation, the covariance
estimator comparison (sample / Ledoit-Wolf / MCD / Marchenko-Pastur), and the
lead-lag (stale-pricing) test against the global equity factors.

Run:  python scripts/01_absorption_ratio.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import warnings
import numpy as np
import pandas as pd

from irdpfn import data_io, absorption as ab, figures as fig, config as C

warnings.filterwarnings("ignore")


def main():
    print("=" * 70)
    print("STEP 01 — Absorption ratio")
    print("=" * 70)

    panel = data_io.load_panel()
    R_f, R_bf, G_b, R_aug = panel.R_f, panel.R_bf, panel.G_b, panel.R_aug
    print(f"R_f {R_f.shape} | R_bf {R_bf.shape} | G_b {G_b.shape} | R_aug {R_aug.shape}")

    # --- three absorption-ratio series ---------------------------------------
    AR_cov = ab.absorption_ratio(R_f, window=C.WINDOW, method="covariance").dropna()
    AR_corr = ab.absorption_ratio(R_f, window=C.WINDOW, method="correlation").dropna()
    AR_aug = ab.absorption_ratio(R_aug, window=C.WINDOW, method="correlation").dropna()

    for name, s in [("covariance R_f", AR_cov),
                    ("correlation R_f", AR_corr),
                    ("correlation R_aug", AR_aug)]:
        print(f"  AR ({name:>18}): mean={s.mean():.4f}  "
              f"range=[{s.min():.4f}, {s.max():.4f}]")

    idx = AR_corr.index.intersection(AR_aug.index)
    rho = np.corrcoef(AR_corr.loc[idx], AR_aug.loc[idx])[0, 1]
    print(f"\ncorr(baseline, augmented) [both correlation-based] = {rho:.3f}")
    print(f"Augmented mean {AR_aug.mean():.4f} < baseline mean {AR_corr.mean():.4f}: "
          f"{AR_aug.mean() < AR_corr.mean()}  (expected True)")

    # --- regularisation materiality (footnote) -------------------------------
    reg = ab.regularisation_check(R_f, window=C.WINDOW)
    print("\nRidge regularisation materiality (baseline R_f, N<window):")
    print(reg.to_string(index=False))

    # --- covariance estimator comparison -------------------------------------
    comp = ab.compare_estimators(R_f, window=C.WINDOW)
    print("\nCovariance estimator comparison (summary over common dates):")
    print(comp.describe().loc[["mean", "std", "min", "max"]].round(4).to_string())

    # --- lead-lag stale-pricing test -----------------------------------------
    r_bar = R_f.mean(axis=1).rename("r_bar")
    print("\nLead-lag corr(r_bar_t, equity_{t-k})  (k>0 => equity leads):")
    for eq in C.EQUITY_COLS:
        ll = ab.lead_lag_corr(r_bar, G_b[eq])
        c0 = ll.loc[ll.lag_k == 0, "corr"].iloc[0]
        c1 = ll.loc[ll.lag_k == 1, "corr"].iloc[0]
        print(f"  {eq:<12} k=0 {c0:+.3f} | k=+1 {c1:+.3f}")

    # --- persist + figures ---------------------------------------------------
    pd.DataFrame({"AR_cov": AR_cov, "AR_corr": AR_corr}).to_csv(
        C.OUTPUTS_DIR / "step01_absorption_ratio.csv")
    AR_aug.to_frame("AR_aug").to_csv(C.OUTPUTS_DIR / "step01_absorption_augmented.csv")
    comp.to_csv(C.OUTPUTS_DIR / "step01_estimator_comparison.csv")
    reg.to_csv(C.OUTPUTS_DIR / "step01_regularisation_check.csv", index=False)

    fig.fig_return_series(R_f, R_bf, G_b)
    fig.fig_baseline_vs_augmented(AR_corr, AR_aug)
    fig.fig_estimator_comparison(comp)
    print(f"\nFigures + tables written to {C.FIGURES_DIR} and {C.OUTPUTS_DIR}")
    print("Step 01 complete.")


if __name__ == "__main__":
    main()
