"""
HMM regime detection on the absorption ratio time series.

Pipeline
--------
A. EM fitting (hmmlearn) for K in {2,...,8}, model selection via BIC.
B. Dual kappa calibration from K=3 and K=5 transition diagonals.
C. Bayesian identifiability under both calibrations (NUTS).
D. Kappa sensitivity grid.
E. EM vs Bayesian agreement at the baseline K=3.
F. Regime characterisation and crisis threshold.
G. Current state, forward probabilities, stationary distribution.
"""

import time
import warnings

import jax
import jax.numpy as jnp
from jax import random

import numpy as np
import pandas as pd

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from hmmlearn.hmm import GaussianHMM

from .config import (
    K_REGIME_BASELINE,
    K_REGIME_RANGE_BIC,
    K_REGIME_RANGE_NUTS,
    KAPPA_SENSITIVITY,
    REGIME_NAMES,
    SEED,
)


# =========================================================
# UTILITIES
# =========================================================
def kappa_from_persistence(p, K, alpha=1.0):
    """
    Invert prior expected self-transition probability to find kappa.

    Prior:  pi_k ~ Dirichlet(alpha * 1 + kappa * e_k)
    E[pi_kk] = (alpha + kappa) / (K*alpha + kappa)
    With alpha=1:  kappa = (K*p - 1) / (1 - p)
    """
    return (K * p - alpha) / (1 - p)


# =========================================================
# STEP A — EM FITTING AND BIC
# =========================================================
def fit_em_hmms(ar_input, K_range=K_REGIME_RANGE_BIC, n_seeds=20, verbose=True):
    """Fit Gaussian HMMs across K range, return best per K + BIC scores."""
    hmm_models, bic_scores = {}, {}
    T = len(ar_input)

    if verbose:
        print(f"{'K':<4} {'log-likelihood':>16} {'n_params':>10} {'BIC':>14}")
        print("-" * 48)

    for K in K_range:
        best_model, best_loglik = None, -np.inf
        for seed in range(n_seeds):
            try:
                m = GaussianHMM(
                    n_components=K, covariance_type="diag",
                    n_iter=1000, tol=1e-6, random_state=seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.fit(ar_input)
                if np.any(m.transmat_.sum(axis=1) < 0.01):
                    continue
                ll = m.score(ar_input)
                if ll > best_loglik:
                    best_loglik, best_model = ll, m
            except Exception:
                continue

        if best_model is None:
            continue

        n_params      = K**2 + 2 * K
        bic           = -2 * best_loglik + n_params * np.log(T)
        hmm_models[K] = best_model
        bic_scores[K] = bic
        if verbose:
            print(f"{K:<4} {best_loglik:>16.2f} {n_params:>10d} {bic:>14.2f}")

    K_bic = min(bic_scores, key=bic_scores.get)
    return hmm_models, bic_scores, K_bic


# =========================================================
# STEP B — KAPPA CALIBRATION FROM EM TRANSITIONS
# =========================================================
def calibrate_kappa(em_model, K):
    """Calibrate kappa from the EM-fitted transition matrix diagonals."""
    diag   = np.diag(em_model.transmat_)
    p_mean = float(diag.mean())
    kappa  = round(kappa_from_persistence(p_mean, K))
    return {
        "K":                K,
        "diag":             diag,
        "persistence_mean": p_mean,
        "kappa_calibrated": kappa,
    }


# =========================================================
# STEP C — STICKY HMM (BAYESIAN, NUTS)
# =========================================================
def sticky_hmm_model(observations, K, kappa=20.0, alpha=1.0,
                     ar_min=0.5, ar_max=1.0):
    """Finite sticky HMM with marginalised forward likelihood."""
    base  = jnp.ones((K, K)) * alpha + kappa * jnp.eye(K)
    pi    = numpyro.sample("pi",    dist.Dirichlet(base))
    mu    = numpyro.sample("mu",    dist.Uniform(ar_min, ar_max).expand([K]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.05).expand([K]))

    init_probs = jnp.ones(K) / K
    log_alpha0 = (jnp.log(init_probs + 1e-30)
                  + dist.Normal(mu, sigma).log_prob(observations[0]))

    def step(log_alpha_prev, obs_t):
        log_alpha = jax.scipy.special.logsumexp(
            log_alpha_prev[:, None] + jnp.log(pi + 1e-30), axis=0,
        ) + dist.Normal(mu, sigma).log_prob(obs_t)
        return log_alpha, log_alpha

    log_alpha_T, _ = jax.lax.scan(step, log_alpha0, observations[1:])
    numpyro.factor("loglik", jax.scipy.special.logsumexp(log_alpha_T))


def run_nuts_hmm(K, ar_data, kappa,
                 n_warmup=1000, n_samples=1000, n_chains=4, seed=SEED,
                 verbose=True):
    """Run NUTS for one (K, kappa) pair."""
    if verbose:
        print(f"  Fitting K = {K}, kappa = {kappa} ...")
    kernel = NUTS(sticky_hmm_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc   = MCMC(kernel,
                  num_warmup=n_warmup, num_samples=n_samples,
                  num_chains=n_chains, chain_method="parallel",
                  progress_bar=verbose)
    t0 = time.time()
    mcmc.run(random.PRNGKey(seed),
             observations=ar_data, K=K, kappa=float(kappa),
             extra_fields=("diverging",))
    if verbose:
        print(f"  K = {K}, kappa = {kappa} completed in "
              f"{(time.time()-t0)/60:.1f} min")
    return mcmc


# =========================================================
# IDENTIFIABILITY DIAGNOSTICS
# =========================================================
def relabel_samples_by_chain(mcmc):
    """Sort posterior samples within each (chain, draw) by emission mean."""
    s   = mcmc.get_samples(group_by_chain=True)
    mu  = np.asarray(s["mu"])
    sig = np.asarray(s["sigma"])
    pi  = np.asarray(s["pi"])
    n_c, n_s, K = mu.shape

    perms = np.argsort(mu, axis=-1)
    mu_s  = np.take_along_axis(mu,  perms, axis=-1)
    sig_s = np.take_along_axis(sig, perms, axis=-1)
    pi_s  = np.zeros_like(pi)
    for c in range(n_c):
        for s_ in range(n_s):
            p = perms[c, s_]
            pi_s[c, s_] = pi[c, s_][np.ix_(p, p)]

    return {"mu": mu_s, "sigma": sig_s, "pi": pi_s}


def gelman_rubin_rhat(samples_3d):
    """Per-parameter R-hat. Input shape: (n_chains, n_samples, K)."""
    n_c, n_s, K = samples_3d.shape
    rhats = np.zeros(K)
    for k in range(K):
        x = samples_3d[:, :, k]
        W = x.var(axis=1, ddof=1).mean()
        B = n_s * x.mean(axis=1).var(ddof=1)
        var_hat  = ((n_s - 1) / n_s) * W + B / n_s
        rhats[k] = np.sqrt(var_hat / W) if W > 0 else np.nan
    return rhats


def identifiability_diagnostics(mcmc, K, kappa):
    """Compute identifiability metrics for one (K, kappa) run."""
    raw      = mcmc.get_samples(group_by_chain=True)
    raw_sum  = numpyro.diagnostics.summary(raw)
    raw_rhat = np.asarray(raw_sum["mu"]["r_hat"])

    rel       = relabel_samples_by_chain(mcmc)
    corr_rhat = gelman_rubin_rhat(rel["mu"])
    n_id      = int((corr_rhat < 1.1).sum())
    mu_sorted = rel["mu"].mean(axis=(0, 1))
    min_gap   = float(np.min(np.diff(mu_sorted)))

    div  = mcmc.get_extra_fields().get("diverging", None)
    n_div = int(np.asarray(div).sum()) if div is not None else 0

    return {
        "K":             K,
        "kappa":         float(kappa),
        "raw_min_rhat":  float(raw_rhat.min()),
        "raw_max_rhat":  float(raw_rhat.max()),
        "corr_min_rhat": float(corr_rhat.min()),
        "corr_max_rhat": float(corr_rhat.max()),
        "n_identified":  n_id,
        "min_mu_gap":    min_gap,
        "n_divergences": n_div,
    }


def run_identifiability_sweep(ar_data, K_list, kappa_list,
                              cached_results=None, verbose=True):
    """
    Run NUTS over a (K, kappa) grid and collect identifiability metrics.

    cached_results : dict {(K, kappa): MCMC} reused across sweeps.
    """
    mcmc_dict   = dict(cached_results) if cached_results else {}
    diagnostics = []
    total       = len(K_list) * len(kappa_list)
    counter     = 0

    for kappa in kappa_list:
        for K in K_list:
            counter += 1
            key = (K, float(kappa))
            if verbose:
                print(f"--- {counter}/{total}: K = {K}, kappa = {kappa} ---")

            if key in mcmc_dict and verbose:
                print("  (cached)")
            elif key not in mcmc_dict:
                mcmc_dict[key] = run_nuts_hmm(K, ar_data, kappa=kappa,
                                              verbose=verbose)

            diagnostics.append(
                identifiability_diagnostics(mcmc_dict[key], K, kappa)
            )

    return mcmc_dict, pd.DataFrame(diagnostics)


# =========================================================
# STEP E — EM vs BAYESIAN AT BASELINE K
# =========================================================
def validate_baseline(em_model, mcmc_baseline,
                      regime_names=REGIME_NAMES):
    """Side-by-side comparison of EM and posterior estimates."""
    K = len(regime_names)

    means_em = em_model.means_.flatten()
    order    = np.argsort(means_em)
    mu_em    = means_em[order]
    sigma_em = np.sqrt(em_model.covars_.flatten()[order])
    pi_em    = em_model.transmat_[np.ix_(order, order)]

    rel      = relabel_samples_by_chain(mcmc_baseline)
    mu_flat  = rel["mu"].reshape(-1, K)
    pi_flat  = rel["pi"].reshape(-1, K, K)
    mu_bayes = mu_flat.mean(axis=0)
    pi_bayes = pi_flat.mean(axis=0)

    rows = []
    for i, name in enumerate(regime_names):
        rows.append({
            "Regime":    name,
            "mu_EM":     mu_em[i],
            "mu_Bayes":  mu_bayes[i],
            "Diff":      abs(mu_em[i] - mu_bayes[i]),
            "Bayes 5%":  np.percentile(mu_flat[:, i], 5),
            "Bayes 95%": np.percentile(mu_flat[:, i], 95),
        })

    return pd.DataFrame(rows), {
        "mu_em": mu_em, "sigma_em": sigma_em, "pi_em": pi_em,
        "mu_bayes": mu_bayes, "pi_bayes": pi_bayes,
        "order": order,
    }


# =========================================================
# STEP F — REGIME CHARACTERISATION
# =========================================================
def characterise_regimes(em_model, ar_input, ar_index,
                         regime_names=REGIME_NAMES):
    """Assign daily regime labels via Viterbi and summarise each regime."""
    means     = em_model.means_.flatten()
    stds      = np.sqrt(em_model.covars_.flatten())
    order     = np.argsort(means)
    states    = em_model.predict(ar_input)
    state_map = {order[k]: regime_names[k] for k in range(len(order))}

    regime_series = pd.Series(
        [state_map[s] for s in states],
        index=ar_index, name="Regime",
    )

    summary = []
    for k, name in enumerate(regime_names):
        s        = order[k]
        days     = (states == s).sum()
        pct      = days / len(states) * 100
        p_self   = em_model.transmat_[s, s]
        duration = 1 / (1 - p_self) if p_self < 1 else np.inf
        summary.append({
            "Regime":      name,
            "Mean":        means[s],
            "Std":         stds[s],
            "Days":        days,
            "Share_pct":   pct,
            "Persistence": p_self,
            "Duration":    duration,
        })

    info = {
        "states": states, "order": order, "state_map": state_map,
        "means": means, "stds": stds,
    }
    return regime_series, pd.DataFrame(summary), info


def compute_thresholds(em_model, N, AR_baseline):
    """Crisis thresholds tau (Mod|High) and tau_1 (Low|Mod) from sorted means."""
    means = np.sort(em_model.means_.flatten())
    tau_1 = (means[0] + means[1]) / 2
    tau   = (means[1] + means[2]) / 2
    rho_1 = (N * tau_1 - 1) / (N - 1)
    rho   = (N * tau   - 1) / (N - 1)
    return {
        "tau_1":       tau_1,
        "tau":         tau,
        "rho_crit_1":  rho_1,
        "rho_crit":    rho,
        "crisis_days": int((AR_baseline > tau).sum()),
        "total_days":  len(AR_baseline),
    }


# =========================================================
# STEP G — CURRENT STATE + FORWARD PROBABILITIES
# =========================================================
def current_state_and_forecast(em_model, regime_info, AR_clean,
                               regime_names=REGIME_NAMES):
    """Current regime, expected duration, multi-horizon forecast,
    stationary distribution."""
    states    = regime_info["states"]
    order     = regime_info["order"]
    state_map = regime_info["state_map"]
    means     = regime_info["means"]
    stds      = regime_info["stds"]

    last_state_raw = int(states[-1])
    last_label     = state_map[last_state_raw]
    p_self         = float(em_model.transmat_[last_state_raw, last_state_raw])
    expected_total = 1 / (1 - p_self) if p_self < 1 else np.inf

    consecutive = 0
    for s in states[::-1]:
        if s == last_state_raw:
            consecutive += 1
        else:
            break

    A = em_model.transmat_
    K = A.shape[0]
    e_current = np.zeros(K)
    e_current[last_state_raw] = 1.0

    horizons = [
        (1,   "t+1 (next day)"),
        (5,   "t+5 (1 week)"),
        (20,  "t+20 (1 month)"),
        (60,  "t+60 (1 quarter)"),
        (120, "t+120 (6 months)"),
    ]

    rows = []
    for h, label_h in horizons:
        prob_h = e_current @ np.linalg.matrix_power(A, h)
        prob_h_sorted = prob_h[order]
        rows.append({"Horizon": label_h,
                     **{regime_names[i]: round(prob_h_sorted[i], 4)
                        for i in range(K)}})

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx              = int(np.argmin(np.abs(eigvals - 1.0)))
    stat             = np.real(eigvecs[:, idx])
    stat             = stat / stat.sum()
    stat_sorted      = stat[order]
    rows.append({"Horizon": "Stationary",
                 **{regime_names[i]: round(stat_sorted[i], 4)
                    for i in range(K)}})

    summary = {
        "last_date":         AR_clean.index[-1],
        "last_label":        last_label,
        "last_ar":           float(AR_clean.iloc[-1]),
        "emission_mean":     means[last_state_raw],
        "emission_std":      stds[last_state_raw],
        "p_self":            p_self,
        "expected_duration": expected_total,
        "consecutive_days":  consecutive,
        "stationary":        stat_sorted,
    }
    return pd.DataFrame(rows), summary
