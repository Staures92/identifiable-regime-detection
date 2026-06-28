"""
Regime detection on the absorption-ratio signal.

Two estimators of the same latent regime structure:

  * EM Gaussian HMM (hmmlearn) for point estimates, BIC model selection,
    regime characterisation, analytical crisis threshold (equal-density
    Gaussian crossing), and forward / stationary regime probabilities;
  * a sticky Bayesian HMM (NumPyro NUTS) with a Dirichlet self-transition
    prior, used for posterior identifiability diagnostics (label-switching
    corrected R-hat) and EM-vs-Bayes validation.

All persistence statements refer to the rolling absorption-ratio signal, not
to independent daily economic states.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from . import config as C


# ===========================================================================
# Sticky-prior utility
# ===========================================================================
def kappa_from_persistence(p: float, K: int, alpha: float = 1.0) -> float:
    """Invert prior expected self-transition: kappa = (K*p - alpha)/(1 - p)."""
    return (K * p - alpha) / (1 - p)


# ===========================================================================
# EM Gaussian HMM
# ===========================================================================
def fit_em_hmms(ar_input: np.ndarray, k_range=C.K_RANGE_BIC, n_seeds: int = 200):
    """Fit Gaussian HMMs across K; return {K: model}, BIC dict, BIC-optimal K.

    n_params = K(K-1) transitions + (K-1) initial + 2K emissions = K^2 + 2K - 1.
    Many restarts keep the log-likelihood monotone non-decreasing in K.
    """
    models, bic, aic = {}, {}, {}
    T = len(ar_input)
    for K in k_range:
        best_model, best_ll = None, -np.inf
        for seed in range(n_seeds):
            try:
                m = GaussianHMM(n_components=K, covariance_type="diag",
                                n_iter=1000, tol=1e-6, random_state=seed)
                m.fit(ar_input)
                if np.any(m.transmat_.sum(axis=1) < 0.01):
                    continue
                ll = m.score(ar_input)
                if ll > best_ll:
                    best_ll, best_model = ll, m
            except Exception:
                continue
        if best_model is None:
            continue
        n_params = K ** 2 + 2 * K - 1
        models[K] = best_model
        aic[K] = -2 * best_ll + 2 * n_params
        bic[K] = -2 * best_ll + n_params * np.log(T)
    K_bic = min(bic, key=bic.get)
    return models, bic, K_bic


def characterise_regimes(em_model, ar_input, ar_index, regime_names=C.REGIME_NAMES):
    """Sort states by emission mean; return Viterbi labels, summary, info dict."""
    means = em_model.means_.flatten()
    stds = np.sqrt(em_model.covars_.flatten())
    order = np.argsort(means)
    states = em_model.predict(ar_input)
    state_map = {order[k]: regime_names[k] for k in range(len(order))}

    regime_series = pd.Series([state_map[s] for s in states],
                              index=ar_index, name="Regime")
    summary = []
    for k, name in enumerate(regime_names):
        s = order[k]
        days = int((states == s).sum())
        p_self = em_model.transmat_[s, s]
        raw_dur = 1 / (1 - p_self) if p_self < 1 else np.inf
        summary.append({
            "Regime": name, "Mean (mu)": means[s], "Std (sigma)": stds[s],
            "Days": days, "Share": f"{days / len(states) * 100:.1f}%",
            "Persistence": round(p_self, 4),
            "Raw_Duration_Days": round(raw_dur, 1),
            "Duration_60D_Windows": round(raw_dur / C.WINDOW, 2),
        })
    info = {"states": states, "order": order, "state_map": state_map,
            "means": means, "stds": stds}
    return regime_series, pd.DataFrame(summary), info


# ===========================================================================
# Analytical crisis threshold (equal-density Gaussian crossing)
# ===========================================================================
def gaussian_crossing(mu1, sigma1, mu2, sigma2) -> float:
    """Root of f1(x)=f2(x) located between the two means."""
    a = 1.0 / sigma2 ** 2 - 1.0 / sigma1 ** 2
    b = -2.0 * mu2 / sigma2 ** 2 + 2.0 * mu1 / sigma1 ** 2
    c = (mu2 ** 2 / sigma2 ** 2 - mu1 ** 2 / sigma1 ** 2
         - 2.0 * np.log(sigma2 / sigma1))
    roots = np.roots([a, b, c])
    roots = roots[np.isreal(roots)].real
    lo, hi = min(mu1, mu2), max(mu1, mu2)
    valid = roots[(roots >= lo) & (roots <= hi)]
    return float(valid[0]) if len(valid) else float((mu1 + mu2) / 2)


def report_threshold_equation(mu_M, sigma_M, mu_H, sigma_H, verbose=True):
    """Print/return the quadratic coefficients for f_M(tau)=f_H(tau)."""
    A = 1.0 / sigma_H ** 2 - 1.0 / sigma_M ** 2
    B = -2.0 * mu_H / sigma_H ** 2 + 2.0 * mu_M / sigma_M ** 2
    C_ = (mu_H ** 2 / sigma_H ** 2 - mu_M ** 2 / sigma_M ** 2
          - 2.0 * np.log(sigma_H / sigma_M))
    roots = np.roots([A, B, C_])
    if verbose:
        print(f"\n[Gaussian crossing]  {A:.6f} tau^2 + {B:.6f} tau + {C_:.6f} = 0")
        for r in roots:
            print(f"  root = {r:.6f}")
    return A, B, C_, roots


def compute_thresholds(em_model, N, ar_series, rho_exact=False, verbose=False):
    """Low|Moderate and Moderate|High boundaries + implied critical rho."""
    means = em_model.means_.flatten()
    sigmas = np.sqrt(em_model.covars_.flatten())
    order = np.argsort(means)
    mu, sd = means[order], sigmas[order]
    tau_1 = gaussian_crossing(mu[0], sd[0], mu[1], sd[1])
    tau = gaussian_crossing(mu[1], sd[1], mu[2], sd[2])
    if verbose:
        print(f"[boundaries] tau_1={tau_1:.4f}  tau={tau:.4f}  "
              f"midpoint={(mu[1] + mu[2]) / 2:.4f}")
    return {"tau_1": tau_1, "tau": tau,
            "rho_crit_1": (N * tau_1 - 1) / (N - 1),
            "rho_crit": (N * tau - 1) / (N - 1),
            "rho_exact": rho_exact,
            "crisis_days": int((ar_series > tau).sum()),
            "total_days": len(ar_series)}


# ===========================================================================
# Current state and forward probabilities
# ===========================================================================
def current_state_and_forecast(em_model, regime_info, ar_clean,
                               regime_names=C.REGIME_NAMES, verbose=True):
    """Current regime, expected duration, multi-horizon forward + stationary."""
    states, order = regime_info["states"], regime_info["order"]
    state_map = regime_info["state_map"]
    last_raw = int(states[-1])
    last_label = state_map[last_raw]
    p_self = float(em_model.transmat_[last_raw, last_raw])
    expected = 1 / (1 - p_self) if p_self < 1 else np.inf
    consecutive = 0
    for s in states[::-1]:
        if s == last_raw:
            consecutive += 1
        else:
            break

    A = em_model.transmat_
    K = A.shape[0]
    e = np.zeros(K); e[last_raw] = 1.0
    rows = []
    for h, label_h in [(1, "t+1 day"), (5, "t+1 week"), (20, "t+1 month")]:
        prob = (e @ np.linalg.matrix_power(A, h))[order]
        rows.append({"Horizon": label_h,
                     **{regime_names[i]: round(prob[i], 4) for i in range(K)}})

    eigval, eigvec = np.linalg.eig(A.T)
    stat = np.real(eigvec[:, int(np.argmin(np.abs(eigval - 1.0)))])
    stat = (stat / stat.sum())[order]
    rows.append({"Horizon": "Stationary",
                 **{regime_names[i]: round(stat[i], 4) for i in range(K)}})

    info = {"last_date": ar_clean.index[-1], "last_label": last_label,
            "p_self": p_self, "expected_duration": expected,
            "consecutive_days": consecutive, "stationary": stat}
    if verbose:
        print(f"[current state {info['last_date'].date()}] {last_label} "
              f"(AR_t={float(ar_clean.iloc[-1]):.4f}, expected {expected:.1f} obs)")
    return pd.DataFrame(rows), info


# ===========================================================================
# One-call EM pipeline on an arbitrary AR series
# ===========================================================================
def run_hmm_pipeline(ar_series, *, label, k_baseline=C.K_BASELINE,
                     k_range_bic=C.K_RANGE_BIC, regime_names=C.REGIME_NAMES,
                     n_seeds=200, rho_exact=False, window=C.WINDOW, verbose=False):
    """EM-HMM + characterisation + threshold on `ar_series` (no NUTS here)."""
    ar_clean = ar_series.replace([np.inf, -np.inf], np.nan).dropna()
    ar_input = ar_clean.values.reshape(-1, 1)

    models, bic, K_bic = fit_em_hmms(ar_input, k_range_bic, n_seeds=n_seeds)
    best_em = models[k_baseline]
    regime_series, regime_summary, regime_info = characterise_regimes(
        best_em, ar_input, ar_clean.index, regime_names)
    thr = compute_thresholds(best_em, C.N, ar_clean, rho_exact=rho_exact,
                             verbose=verbose)
    order = regime_info["order"]
    return {
        "label": label, "rho_exact": rho_exact, "K_bic": K_bic,
        "K_baseline": k_baseline, "em": best_em, "hmm_models": models,
        "bic_scores": bic, "regime_series": regime_series,
        "regime_summary": regime_summary, "regime_info": regime_info,
        "mu": regime_info["means"][order], "sigma": regime_info["stds"][order],
        "persistence": np.diag(best_em.transmat_)[order],
        "tau": thr["tau"], "tau_1": thr["tau_1"],
        "rho_crit": thr["rho_crit"], "rho_crit_1": thr["rho_crit_1"],
        "crisis_days": thr["crisis_days"], "total_days": thr["total_days"],
        "ar_clean": ar_clean, "regime_names": regime_names, "window": window,
    }


def compare_hmm_robustness(res_a, res_b, verbose=True):
    """Single side-by-side EM-HMM comparison (regimes, threshold, agreement)."""
    from sklearn.metrics import adjusted_rand_score
    rn = res_a["regime_names"]
    la, lb = res_a["label"], res_b["label"]
    rows = []
    for i, name in enumerate(rn):
        rows.append({"Regime": name,
                     f"mu_{la[:4]}": round(res_a["mu"][i], 4),
                     f"mu_{lb[:4]}": round(res_b["mu"][i], 4),
                     f"sig_{la[:4]}": round(res_a["sigma"][i], 4),
                     f"sig_{lb[:4]}": round(res_b["sigma"][i], 4),
                     f"pp_{la[:4]}": round(res_a["persistence"][i], 4),
                     f"pp_{lb[:4]}": round(res_b["persistence"][i], 4)})
    table = pd.DataFrame(rows)

    idx = res_a["regime_series"].index.intersection(res_b["regime_series"].index)
    a = res_a["regime_series"].loc[idx].astype("category").cat.codes
    b = res_b["regime_series"].loc[idx].astype("category").cat.codes
    ari = adjusted_rand_score(a, b)
    agree = float((res_a["regime_series"].loc[idx].values ==
                   res_b["regime_series"].loc[idx].values).mean())
    ct = pd.crosstab(res_a["regime_series"].loc[idx].rename(la),
                     res_b["regime_series"].loc[idx].rename(lb))
    if verbose:
        print(f"\nHMM robustness: {la} vs {lb} "
              f"(BIC K {res_a['K_bic']} / {res_b['K_bic']})")
        print(table.to_string(index=False))
        print(f"tau: {la}={res_a['tau']:.4f} | {lb}={res_b['tau']:.4f}")
        print(f"daily agreement {agree * 100:.1f}% | ARI {ari:.4f}")
    return {"table": table, "crosstab": ct, "ari": ari, "agreement": agree}


def window_robustness(R_f, windows=C.WINDOWS, n_seeds=50, absorption_fn=None):
    """Refit the EM pipeline across rolling-window lengths; summary frame."""
    if absorption_fn is None:
        from .absorption import absorption_ratio as absorption_fn
    results = {}
    for w in windows:
        ar = absorption_fn(R_f, window=w, method="covariance").dropna()
        results[w] = run_hmm_pipeline(ar, label=f"w{w}", rho_exact=False,
                                      n_seeds=n_seeds, window=w)
    summary = pd.DataFrame([
        {"Window": w, "Mean AR": r["ar_clean"].mean(),
         "Std AR": r["ar_clean"].std(), "BIC K": r["K_bic"],
         "Threshold": r["tau"],
         "High-Concentration (%)": 100 * r["crisis_days"] / r["total_days"]}
        for w, r in results.items()
    ]).sort_values("Window")
    return results, summary


# ===========================================================================
# Overlapping-window diagnostics (Reviewer Comment 3)
# ===========================================================================
def overlap_acf(ar_series, window=C.WINDOW, n_lags=180):
    """Empirical AR autocorrelation vs the overlap-only benchmark."""
    from statsmodels.tsa.stattools import acf
    ar = ar_series.replace([np.inf, -np.inf], np.nan).dropna().values
    emp = acf(ar, nlags=n_lags, fft=True)
    lags = np.arange(n_lags + 1)
    mech = np.maximum(0.0, (window - lags) / window)
    beyond = emp[window:].mean()
    return pd.DataFrame({"lag": lags, "empirical": emp,
                         "overlap_benchmark": mech}), beyond


def overlap_free_regime_check(ar_full, res_ref, window=C.WINDOW,
                              K=C.K_BASELINE, n_seeds=20):
    """Refit regimes on disjoint subsamples (every window-th point)."""
    AR = ar_full.replace([np.inf, -np.inf], np.nan).dropna()

    def fit_one(X):
        best, ll = None, -np.inf
        for s in range(n_seeds):
            try:
                m = GaussianHMM(n_components=K, covariance_type="diag",
                                n_iter=100, tol=1e-4, random_state=s).fit(X)
                sc = m.score(X)
                if sc > ll:
                    ll, best = sc, m
            except Exception:
                continue
        if best is None:
            return None, None
        mu = best.means_.flatten(); sd = np.sqrt(best.covars_.flatten())
        o = np.argsort(mu)
        return mu[o], sd[o]

    mus, taus = [], []
    for off in range(window):
        sub = AR.iloc[off::window]
        if len(sub) < 8 * K:
            continue
        mu, sd = fit_one(sub.values.reshape(-1, 1))
        if mu is None:
            continue
        mus.append(mu)
        taus.append(gaussian_crossing(mu[1], sd[1], mu[2], sd[2]))
    mus, taus = np.array(mus), np.array(taus)
    out = pd.DataFrame({"Regime": C.REGIME_NAMES,
                        "mu_overlapping": np.round(np.asarray(res_ref["mu"]), 4),
                        "mu_nonoverlap_mean": np.round(mus.mean(0), 4),
                        "mu_nonoverlap_sd": np.round(mus.std(0), 4)})
    return out, res_ref["tau"], taus


# ===========================================================================
# Sticky Bayesian HMM (NumPyro)  --  imported lazily
# ===========================================================================
def sticky_hmm_model(observations, K, kappa=20.0, alpha=1.0,
                     ar_min=0.5, ar_max=1.0):
    """Finite sticky HMM with marginalised forward likelihood."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    base = jnp.ones((K, K)) * alpha + kappa * jnp.eye(K)
    pi = numpyro.sample("pi", dist.Dirichlet(base))
    mu = numpyro.sample("mu", dist.Uniform(ar_min, ar_max).expand([K]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.05).expand([K]))

    init = jnp.ones(K) / K
    log_a0 = jnp.log(init + 1e-30) + dist.Normal(mu, sigma).log_prob(observations[0])

    def step(log_a_prev, obs_t):
        log_a = jax.scipy.special.logsumexp(
            log_a_prev[:, None] + jnp.log(pi + 1e-30), axis=0)
        log_a = log_a + dist.Normal(mu, sigma).log_prob(obs_t)
        return log_a, log_a

    log_aT, _ = jax.lax.scan(step, log_a0, observations[1:])
    numpyro.factor("loglik", jax.scipy.special.logsumexp(log_aT))


def run_nuts(K, observations, kappa=20.0, alpha=1.0,
             mcmc_settings=None, seed=C.SEED):
    """Run NUTS for the sticky Bayesian HMM, returning the fitted MCMC."""
    import jax
    from numpyro.infer import NUTS, MCMC
    if mcmc_settings is None:
        mcmc_settings = C.MCMC_FAST
    kernel = NUTS(sticky_hmm_model, target_accept_prob=0.90)
    mcmc = MCMC(kernel, num_warmup=mcmc_settings["num_warmup"],
                num_samples=mcmc_settings["num_samples"],
                num_chains=mcmc_settings["num_chains"], progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), observations=observations,
             K=K, kappa=kappa, alpha=alpha)
    return mcmc


def _relabel_by_chain(mcmc):
    """Sort posterior samples within each (chain, draw) by emission mean."""
    s = mcmc.get_samples(group_by_chain=True)
    mu = np.asarray(s["mu"]); sigma = np.asarray(s["sigma"]); pi = np.asarray(s["pi"])
    nc, ns, K = mu.shape
    perms = np.argsort(mu, axis=-1)
    mu_s = np.take_along_axis(mu, perms, axis=-1)
    sigma_s = np.take_along_axis(sigma, perms, axis=-1)
    pi_s = np.zeros_like(pi)
    for c in range(nc):
        for d in range(ns):
            p = perms[c, d]
            pi_s[c, d] = pi[c, d][np.ix_(p, p)]
    return {"mu": mu_s, "sigma": sigma_s, "pi": pi_s}


def _gelman_rubin(samples_3d):
    """Per-parameter R-hat for an array shaped (chains, samples, K)."""
    nc, ns, K = samples_3d.shape
    rhats = np.zeros(K)
    for k in range(K):
        x = samples_3d[:, :, k]
        W = x.var(axis=1, ddof=1).mean()
        B = ns * x.mean(axis=1).var(ddof=1)
        var_hat = ((ns - 1) / ns) * W + B / ns
        rhats[k] = np.sqrt(var_hat / W) if W > 0 else np.nan
    return rhats


def identifiability_diagnostics(mcmc, K, kappa):
    """Raw vs label-switching-corrected R-hat and identified-state count."""
    import numpyro
    raw = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    raw_rhat = np.asarray(raw["mu"]["r_hat"])
    rel = _relabel_by_chain(mcmc)
    corr_rhat = _gelman_rubin(rel["mu"])
    diverging = mcmc.get_extra_fields().get("diverging", None)
    n_div = int(np.asarray(diverging).sum()) if diverging is not None else 0
    return {"K": K, "kappa": float(kappa),
            "raw_min_rhat": float(raw_rhat.min()), "raw_max_rhat": float(raw_rhat.max()),
            "corr_min_rhat": float(corr_rhat.min()), "corr_max_rhat": float(corr_rhat.max()),
            "n_identified": int((corr_rhat < 1.1).sum()),
            "min_mu_gap": float(np.min(np.diff(rel["mu"].mean(axis=(0, 1))))),
            "n_divergences": n_div}


def identifiability_sweep(ar_data, K_list, kappa_list, cached=None):
    """Run NUTS for each (K, kappa); reuse a {(K,kappa): mcmc} cache."""
    cache = dict(cached) if cached else {}
    diags = []
    for kappa in kappa_list:
        for K in K_list:
            key = (K, float(kappa))
            if key not in cache:
                cache[key] = run_nuts(K, ar_data, kappa=kappa)
            diags.append(identifiability_diagnostics(cache[key], K, kappa))
    return cache, pd.DataFrame(diags)


def calibrate_kappa(em_model, K):
    """Calibrate kappa from EM transition-matrix diagonals."""
    diag = np.diag(em_model.transmat_)
    p_mean = float(diag.mean())
    return {"K": K, "diag": diag, "persistence_mean": p_mean,
            "kappa_calibrated": round(kappa_from_persistence(p_mean, K))}


def validate_baseline(em_model, mcmc_baseline, regime_names=C.REGIME_NAMES):
    """Compare EM point estimates with Bayesian posterior means at baseline K."""
    K = len(regime_names)
    means_em = em_model.means_.flatten()
    order = np.argsort(means_em)
    mu_em = means_em[order]
    rel = _relabel_by_chain(mcmc_baseline)
    mu_flat = rel["mu"].reshape(-1, K)
    mu_bayes = mu_flat.mean(axis=0)
    rows = []
    for i, name in enumerate(regime_names):
        rows.append({"Regime": name, "mu_EM": mu_em[i], "mu_Bayes": mu_bayes[i],
                     "Diff": abs(mu_em[i] - mu_bayes[i]),
                     "Bayes 5%": np.percentile(mu_flat[:, i], 5),
                     "Bayes 95%": np.percentile(mu_flat[:, i], 95)})
    return pd.DataFrame(rows)
