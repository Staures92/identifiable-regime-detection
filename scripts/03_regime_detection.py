"""
Step 03 — Regime detection on the absorption ratio.

Fits Gaussian EM-HMMs across K, reports the BIC-optimal K, and characterises the
K = 3 baseline (Low / Moderate / High concentration). The crisis threshold tau
is the equal-density Gaussian crossing between the Moderate and High emissions.
Adds the robustness battery: correlation-AR vs covariance-AR regimes,
window-length sensitivity, the overlap-only ACF benchmark, and the decisive
overlap-free (disjoint-subsample) regime check.

The sticky Bayesian HMM identifiability sweep (NUTS) is slow and is gated behind
``--bayes``. By default it is skipped so the script runs offline in minutes.

Run:  python scripts/03_regime_detection.py [--bayes] [--n-seeds 200]
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import argparse
import warnings
import numpy as np

from irdpfn import data_io, absorption as ab, regime as rg, figures as fig, config as C

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bayes", action="store_true",
                   help="run the NUTS identifiability sweep (slow)")
    p.add_argument("--n-seeds", type=int, default=8,
                   help="EM restarts per K (paper uses 200)")
    args = p.parse_args()

    print("=" * 70)
    print("STEP 03 — Regime detection")
    print("=" * 70)

    panel = data_io.load_panel()
    AR_cov = ab.absorption_ratio(panel.R_f, window=C.WINDOW, method="covariance").dropna()
    AR_corr = ab.absorption_ratio(panel.R_f, window=C.WINDOW, method="correlation").dropna()

    # --- baseline EM-HMM -----------------------------------------------------
    res = rg.run_hmm_pipeline(AR_cov, label="covariance", n_seeds=args.n_seeds,
                              rho_exact=False)
    print(f"\nBIC-optimal K = {res['K_bic']}   (baseline analysis fixes K = {C.K_BASELINE})")
    print(res["regime_summary"].to_string(index=False))
    print(f"\ntau_1 (Low|Moderate)  = {res['tau_1']:.4f}")
    print(f"tau   (Moderate|High) = {res['tau']:.4f}")
    print(f"Crisis observations   = {res['crisis_days']}/{res['total_days']} "
          f"({100*res['crisis_days']/res['total_days']:.1f}%)")
    rg.report_threshold_equation(res["mu"][1], res["sigma"][1],
                                 res["mu"][2], res["sigma"][2])

    # --- current state + forward probabilities -------------------------------
    forecast, _ = rg.current_state_and_forecast(res["em"], res["regime_info"],
                                                res["ar_clean"])

    # --- correlation-AR robustness -------------------------------------------
    res_corr = rg.run_hmm_pipeline(AR_corr, label="correlation",
                                   n_seeds=args.n_seeds, rho_exact=True)
    rg.compare_hmm_robustness(res, res_corr)

    # --- window-length robustness --------------------------------------------
    win_seeds = min(args.n_seeds, 4)
    _, win = rg.window_robustness(panel.R_f, n_seeds=win_seeds)
    print("\nWindow-length robustness:")
    print(win.round(4).to_string(index=False))

    # --- overlap diagnostics -------------------------------------------------
    acf_df, beyond = rg.overlap_acf(res["ar_clean"])
    print(f"\nMean |ACF| beyond the window horizon: {beyond:.4f}")
    rg.overlap_free_regime_check(res["ar_clean"], res)

    # --- optional Bayesian identifiability sweep -----------------------------
    if args.bayes:
        print("\n" + "=" * 70)
        print("Sticky Bayesian HMM — NUTS identifiability sweep")
        print("=" * 70)
        ar_data = np.asarray(res["ar_clean"].values, dtype=np.float32)
        k3 = rg.calibrate_kappa(res["hmm_models"][C.K_BASELINE], C.K_BASELINE)
        print(f"Calibrated kappa (K=3): {k3['kappa_calibrated']}")
        cache, diag = rg.identifiability_sweep(
            ar_data, C.K_RANGE_NUTS, [float(k3["kappa_calibrated"])])
        print(diag.to_string(index=False))
        diag.to_csv(C.OUTPUTS_DIR / "step03_identifiability.csv", index=False)
        mcmc0 = cache[(C.K_BASELINE, float(k3["kappa_calibrated"]))]
        val, _ = rg.validate_baseline(res["em"], mcmc0)
        print("\nEM vs Bayesian baseline:")
        print(val.to_string(index=False))
        val.to_csv(C.OUTPUTS_DIR / "step03_em_vs_bayes.csv", index=False)

    # --- persist + figures ---------------------------------------------------
    res["regime_series"].to_csv(C.OUTPUTS_DIR / "step03_regime_labels.csv")
    res["regime_summary"].to_csv(C.OUTPUTS_DIR / "step03_regime_summary.csv", index=False)
    forecast.to_csv(C.OUTPUTS_DIR / "step03_forward_probabilities.csv", index=False)
    win.to_csv(C.OUTPUTS_DIR / "step03_window_robustness.csv", index=False)

    fig.fig_ar_benchmarks(res["ar_clean"], res["regime_series"], res["tau"],
                          panel.R_bf, panel.G_b)
    fig.fig_emission_distributions(res)
    fig.fig_regime_classification(res)
    fig.fig_transition_matrix(res["em"])
    fig.fig_acf(acf_df)
    print(f"\nFigures + tables written to {C.FIGURES_DIR} and {C.OUTPUTS_DIR}")
    print("Step 03 complete.")


if __name__ == "__main__":
    main()
