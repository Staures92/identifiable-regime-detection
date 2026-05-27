"""
Step 03 — HMM regime detection on AR_t.

Pipeline:
    A. EM fitting (hmmlearn), BIC selection
    B. Dual kappa calibration from K=3 and K=5 EM transitions
    C. Bayesian identifiability under both calibrations
    D. Kappa sensitivity grid
    E. EM vs Bayesian validation at baseline K
    F. Regime characterisation and crisis thresholds
    G. Current state, forward probabilities, stationary distribution

By default this script runs everything. Set `RUN_BAYESIAN = False` at the
top to skip the NUTS sweeps (steps C, D, E) and only run the EM-based
analysis. This is useful when developing or when JAX/NumPyro isn't
available.

Outputs:
    step03_bic.csv
    step03_kappa_calibration.csv
    step03_identifiability_kappa_K3.csv     (if Bayesian on)
    step03_identifiability_kappa_K5.csv     (if Bayesian on)
    step03_kappa_sensitivity.csv            (if Bayesian on)
    step03_em_vs_bayesian_validation.csv    (if Bayesian on)
    step03_regime_summary.csv
    step03_regime_labels.csv
    step03_thresholds.csv
    step03_forward_probabilities.csv
    step03_current_state.csv

Figures:
    fig03_ar_benchmarks_regimes.pdf
    fig07_emission_distributions.pdf
    fig11_risk_decomposition.pdf            (re-rendered with regimes)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---- toggle ----
RUN_BAYESIAN = False
# RUN_BAYESIAN = True

from irdpfn.absorption import compute_absorption_ratio, compute_risk_decomposition
from irdpfn.config import (
    K_REGIME_BASELINE,
    K_REGIME_RANGE_NUTS,
    KAPPA_SENSITIVITY,
    N_FUNDS,
    OUTPUTS_DIR,
    REGIME_NAMES,
)
from irdpfn.data_io import load_and_align
from irdpfn.figures import (
    fig03_ar_benchmarks_regimes,
    fig07_emission_distributions,
    fig11_risk_decomposition,
)
from irdpfn.regime import (
    calibrate_kappa,
    characterise_regimes,
    compute_thresholds,
    current_state_and_forecast,
    fit_em_hmms,
    run_identifiability_sweep,
    validate_baseline,
)


def main():
    # 1. Load data and compute AR_t
    df, R_f, R_bf, R_bg, _ = load_and_align()
    AR_baseline = compute_absorption_ratio(R_f).dropna()
    AR_clean    = AR_baseline.replace([np.inf, -np.inf], np.nan).dropna()
    ar_input    = AR_clean.values.reshape(-1, 1)

    # 2. Step A — EM and BIC
    print("\n" + "=" * 60)
    print("STEP A — EM fitting and BIC selection")
    print("=" * 60)
    hmm_models, bic_scores, K_bic = fit_em_hmms(ar_input)

    ll_best  = max(hmm_models[k].score(ar_input) for k in hmm_models)
    bic_best = min(bic_scores.values())
    bic_rows = []
    for k in sorted(bic_scores):
        ll_k = hmm_models[k].score(ar_input)
        bic_rows.append({
            "K":               k,
            "Log_likelihood":  round(ll_k, 2),
            "Delta_LL":        round(ll_k - ll_best, 2),
            "n_params":        k**2 + 2 * k,
            "BIC":             round(bic_scores[k], 2),
            "Delta_BIC":       round(bic_scores[k] - bic_best, 2),
            "BIC_optimal":     k == K_bic,
            "Theory_baseline": k == K_REGIME_BASELINE,
        })
    bic_df = pd.DataFrame(bic_rows)
    bic_df.to_csv(OUTPUTS_DIR / "step03_bic.csv", index=False)
    print(f"\nBIC-optimal: K = {K_bic}")
    print(f"Theory baseline: K = {K_REGIME_BASELINE}")
    print(bic_df.to_string(index=False))

    # 3. Step B — Kappa calibration
    print("\n" + "=" * 60)
    print("STEP B — Kappa calibration from EM transitions")
    print("=" * 60)
    k3 = calibrate_kappa(hmm_models[3], 3)
    k5 = calibrate_kappa(hmm_models[5], 5) if 5 in hmm_models else None
    pd.DataFrame([
        {kk: vv for kk, vv in k3.items() if kk != "diag"},
        *([] if k5 is None else
          [{kk: vv for kk, vv in k5.items() if kk != "diag"}]),
    ]).to_csv(OUTPUTS_DIR / "step03_kappa_calibration.csv", index=False)
    print(f"kappa from K=3: {k3['kappa_calibrated']}")
    if k5 is not None:
        print(f"kappa from K=5: {k5['kappa_calibrated']}")

    # 4. Bayesian steps (optional)
    mcmc_cache = {}
    if RUN_BAYESIAN:
        import jax.numpy as jnp
        ar_data = jnp.array(AR_clean.values, dtype=jnp.float32)

        print("\n" + "=" * 60)
        print("STEP C — Bayesian identifiability under dual calibration")
        print("=" * 60)

        mcmc_cache, diag_K3 = run_identifiability_sweep(
            ar_data, K_REGIME_RANGE_NUTS, [float(k3["kappa_calibrated"])],
        )
        diag_K3.to_csv(
            OUTPUTS_DIR / "step03_identifiability_kappa_K3.csv", index=False,
        )

        if k5 is not None:
            mcmc_cache, diag_K5 = run_identifiability_sweep(
                ar_data, K_REGIME_RANGE_NUTS, [float(k5["kappa_calibrated"])],
                cached_results=mcmc_cache,
            )
            diag_K5.to_csv(
                OUTPUTS_DIR / "step03_identifiability_kappa_K5.csv",
                index=False,
            )

        print("\n" + "=" * 60)
        print("STEP D — Kappa sensitivity")
        print("=" * 60)
        all_kappas = sorted(set(
            KAPPA_SENSITIVITY
            + [float(k3["kappa_calibrated"])]
            + ([float(k5["kappa_calibrated"])] if k5 is not None else [])
        ))
        mcmc_cache, sens = run_identifiability_sweep(
            ar_data, K_REGIME_RANGE_NUTS, all_kappas,
            cached_results=mcmc_cache,
        )
        sens.to_csv(OUTPUTS_DIR / "step03_kappa_sensitivity.csv", index=False)

        print("\n" + "=" * 60)
        print(f"STEP E — EM vs Bayesian at K = {K_REGIME_BASELINE}")
        print("=" * 60)
        baseline_key = (K_REGIME_BASELINE, float(k3["kappa_calibrated"]))
        if baseline_key in mcmc_cache:
            valid_df, _ = validate_baseline(
                hmm_models[K_REGIME_BASELINE],
                mcmc_cache[baseline_key],
            )
            valid_df.to_csv(
                OUTPUTS_DIR / "step03_em_vs_bayesian_validation.csv",
                index=False,
            )
            print(valid_df.to_string(index=False))
    else:
        print("\n[Bayesian sweep skipped — set RUN_BAYESIAN = True to enable]")

    # 5. Step F — Regime characterisation
    print("\n" + "=" * 60)
    print("STEP F — Regime characterisation and crisis threshold")
    print("=" * 60)
    em_baseline = hmm_models[K_REGIME_BASELINE]
    regime_series, regime_summary, regime_info = characterise_regimes(
        em_baseline, ar_input, AR_clean.index,
    )
    regime_series.to_csv(OUTPUTS_DIR / "step03_regime_labels.csv")
    regime_summary.to_csv(
        OUTPUTS_DIR / "step03_regime_summary.csv", index=False,
    )
    print(regime_summary.to_string(index=False))

    thr = compute_thresholds(em_baseline, N_FUNDS, AR_baseline)
    pd.DataFrame([thr]).to_csv(
        OUTPUTS_DIR / "step03_thresholds.csv", index=False,
    )
    print(f"\ntau   = {thr['tau']:.4f}   rho_crit  = {thr['rho_crit']:.4f}")
    print(f"tau_1 = {thr['tau_1']:.4f}   rho_crit_1 = {thr['rho_crit_1']:.4f}")
    print(f"Crisis days (AR > tau): {thr['crisis_days']}/{thr['total_days']}")

    # 6. Step G — Current state and forecast
    print("\n" + "=" * 60)
    print("STEP G — Current state and forward probabilities")
    print("=" * 60)
    forecast_df, summary = current_state_and_forecast(
        em_baseline, regime_info, AR_clean,
    )
    forecast_df.to_csv(
        OUTPUTS_DIR / "step03_forward_probabilities.csv", index=False,
    )
    pd.DataFrame([{
        k: (v if not hasattr(v, "tolist") else str(v.tolist()))
        for k, v in summary.items()
    }]).to_csv(OUTPUTS_DIR / "step03_current_state.csv", index=False)
    print(f"Current regime: {summary['last_label']}  "
          f"(AR_t = {summary['last_ar']:.4f})")
    print(f"Expected total duration: {summary['expected_duration']:.1f} days")
    print(f"Consecutive days in regime: {summary['consecutive_days']}")
    print("\nForward probabilities:")
    print(forecast_df.to_string(index=False))

    # 7. Figures
    fig03_ar_benchmarks_regimes(AR_clean, R_bf, R_bg,
                                regime_series, tau=thr["tau"])

    means = em_baseline.means_.flatten()
    stds  = np.sqrt(em_baseline.covars_.flatten())
    order = np.argsort(means)
    regime_params = {
        REGIME_NAMES[i]: (means[order[i]], stds[order[i]])
        for i in range(len(REGIME_NAMES))
    }
    fig07_emission_distributions(regime_params, thr["tau"], thr["tau_1"])

    risk_decomp = compute_risk_decomposition(R_f)
    fig11_risk_decomposition(risk_decomp, regime_series, AR_clean.index)

    print("\nStep 03 complete.")


if __name__ == "__main__":
    main()
