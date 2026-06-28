"""
Step 04 — Cross-step diagnostics.

Ties the absorption ratio, the HMM regimes, and the DTW clusters together:
  * Pearson / Spearman / Kendall correlations of AR_t with the interpretive
    series (MAAR and the five global factors);
  * a regime-conditional OLS with HAC errors, where a significant interaction
    b3 means the AR-to-factor relationship is regime-dependent;
  * the augmented-AR crisis/calm agreement with its permutation test;
  * per-cluster mean absolute active return amplified in crisis vs normal.

Regimes are recomputed here; clusters are read from ``outputs/step02_clusters.csv``
when present, otherwise a fast hierarchical-only clustering is run.

Run:  python scripts/04_diagnostics.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import warnings
import numpy as np
import pandas as pd

from irdpfn import (data_io, absorption as ab, regime as rg, clustering as cl,
                    diagnostics as dg, figures as fig, config as C)

warnings.filterwarnings("ignore")


def _load_clusters(panel):
    path = C.OUTPUTS_DIR / "step02_clusters.csv"
    if path.exists():
        return pd.read_csv(path)
    res = cl.run_clustering(panel.R_f, panel.R_bf,
                            robustness=False, noncircularity=False)
    return cl.cluster_frame(res, which="hier")


def main():
    print("=" * 70)
    print("STEP 04 — Cross-step diagnostics")
    print("=" * 70)

    panel = data_io.load_panel()
    AR_cov = ab.absorption_ratio(panel.R_f, window=C.WINDOW, method="covariance").dropna()
    AR_corr = ab.absorption_ratio(panel.R_f, window=C.WINDOW, method="correlation").dropna()
    AR_aug = ab.absorption_ratio(panel.R_aug, window=C.WINDOW, method="correlation").dropna()

    res = rg.run_hmm_pipeline(AR_cov, label="covariance", n_seeds=8, rho_exact=False)
    regimes = res["regime_series"]

    # --- correlation battery -------------------------------------------------
    interp = dg.build_interp_frame(AR_cov, panel.R_f, panel.R_bf, panel.G_b, regimes)
    corr_tbl = dg.correlation_table(interp)
    print("\nCorrelation of AR_t with interpretive series:")
    print(corr_tbl.to_string(index=False))

    # --- regime-conditional OLS ----------------------------------------------
    ols_tbl, fitted = dg.regime_conditional_ols(interp)
    print("\nRegime-conditional OLS (HAC); b3 = AR x D_High interaction:")
    print(ols_tbl.to_string(index=False))

    # --- augmented agreement + permutation -----------------------------------
    agree = dg.augmented_agreement(AR_corr, AR_aug)
    perm = dg.permutation_gap_test(agree)
    print(f"\nAugmented-AR agreement: rho_overall={agree['rho_overall']:.3f}  "
          f"crisis={agree['rho_crisis']:.3f}  calm={agree['rho_calm']:.3f}")
    print(f"crisis-calm gap = {perm['observed_gap']:+.3f}  "
          f"(permutation p = {perm['p_perm']:.3f})")

    # --- cluster MAAR amplification ------------------------------------------
    cluster_df = _load_clusters(panel)
    maar = dg.cluster_maar_amplification(panel.R_f, panel.R_bf, cluster_df,
                                         regimes, res["ar_clean"].index)
    print("\nPer-cluster MAAR amplification (crisis / normal):")
    print(maar.round(6).to_string(index=False))

    # --- persist + figures ---------------------------------------------------
    corr_tbl.to_csv(C.OUTPUTS_DIR / "step04_correlations.csv", index=False)
    ols_tbl.to_csv(C.OUTPUTS_DIR / "step04_regime_ols.csv", index=False)
    maar.to_csv(C.OUTPUTS_DIR / "step04_cluster_maar.csv", index=False)

    fig.fig_scatter_correlations(interp, res["tau"])
    fig.fig_regime_conditional_scatter(interp, fitted, res["tau"])
    fig.fig_cluster_maar(maar)
    print(f"\nFigures + tables written to {C.FIGURES_DIR} and {C.OUTPUTS_DIR}")
    print("Step 04 complete.")


if __name__ == "__main__":
    main()
