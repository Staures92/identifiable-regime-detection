"""
Absorption ratio and its robustness checks.

AR_t = lambda_1(M_t) / trace(M_t) over a rolling window, where M_t is either
the sample covariance (paper baseline) or the sample correlation (scale-free
robustness). A single function serves both; alternative shrinkage / robust /
random-matrix estimators live alongside for Reviewer Comment 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Canonical absorption ratio
# ---------------------------------------------------------------------------
def absorption_ratio(returns_df: pd.DataFrame, window: int = C.WINDOW,
                     method: str = "covariance") -> pd.Series:
    """
    Rolling AR_t.

    method='covariance'  : M_t = sample covariance (paper baseline). High-vol
                           cohorts dominate the trace; AR_t approximates the
                           mean correlation only loosely.
    method='correlation' : M_t = sample correlation (each window standardised
                           to unit variance), so trace = N exactly and
                           rho_bar = (N*AR_t - 1)/(N - 1) is exact.

    A ridge (+RIDGE_EPS*I) is added only when N > window (the rank-deficient
    augmented matrix); the full-rank baseline R_f is left untouched.
    """
    values = returns_df.values.astype(float)
    T, N = values.shape
    ar = np.full(T, np.nan)

    for t in range(window - 1, T):
        W = values[t - window + 1: t + 1]
        if not np.all(np.isfinite(W)):
            continue
        Wc = W - W.mean(axis=0)
        if method == "correlation":
            sd = W.std(axis=0, ddof=1)
            sd[sd < 1e-12] = 1.0
            Wc = Wc / sd
        elif method != "covariance":
            raise ValueError("method must be 'covariance' or 'correlation'")

        M = (Wc.T @ Wc) / (window - 1)
        if N > window:
            M += C.RIDGE_EPS * np.eye(N)

        eig = np.maximum(np.linalg.eigvalsh(M), 0.0)
        total = eig.sum()
        if total > 0:
            ar[t] = eig[-1] / total

    return pd.Series(ar, index=returns_df.index, name=f"AR_{method}")


def rho_from_ar(ar: np.ndarray | float, N: int) -> np.ndarray | float:
    """Exact mean-correlation implied by a correlation-based AR_t."""
    return (N * np.asarray(ar) - 1.0) / (N - 1.0)


# ---------------------------------------------------------------------------
# Regularisation materiality (footnote 1)
# ---------------------------------------------------------------------------
def regularisation_check(R_f: pd.DataFrame, window: int = C.WINDOW) -> pd.DataFrame:
    """Confirm the ridge is immaterial on the full-rank baseline (N < window)."""
    rows = []
    N = R_f.shape[1]
    for label, start in [("Early window", 0),
                         ("Mid window", len(R_f) // 2),
                         ("Late window", len(R_f) - window)]:
        W = R_f.values[start: start + window]
        Wc = W - W.mean(axis=0)
        Sigma = (Wc.T @ Wc) / (window - 1)
        ev_u = np.linalg.eigvalsh(Sigma)
        ev_r = np.linalg.eigvalsh(Sigma + C.RIDGE_EPS * np.eye(N))
        ar_u = ev_u[-1] / ev_u.sum()
        ar_r = ev_r[-1] / ev_r.sum()
        rows.append({"Window": label, "AR_unreg": round(ar_u, 6),
                     "AR_reg": round(ar_r, 6),
                     "Difference": round(abs(ar_u - ar_r), 6)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alternative covariance estimators (Reviewer Comment 6)
# ---------------------------------------------------------------------------
def _mp_filter_covariance(W: np.ndarray) -> np.ndarray:
    """Marchenko-Pastur eigenvalue clipping on the standardised correlation."""
    T, N = W.shape
    sd = W.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    Wstd = (W - W.mean(axis=0)) / sd
    Cmat = (Wstd.T @ Wstd) / (T - 1)
    eig, vec = np.linalg.eigh(Cmat)
    lam_plus = (1.0 + np.sqrt(N / T)) ** 2
    noise = eig <= lam_plus
    if noise.any():
        eig = np.where(noise, eig[noise].mean(), eig)
    Cf = vec @ np.diag(eig) @ vec.T
    np.fill_diagonal(Cf, 1.0)
    return Cf * np.outer(sd, sd)


def absorption_ratio_estimator(returns_df: pd.DataFrame, window: int = C.WINDOW,
                               estimator: str = "sample") -> pd.Series:
    """AR_t under {'sample','ledoit_wolf','mcd','mp_filtered'} covariance."""
    from sklearn.covariance import LedoitWolf, MinCovDet
    values = returns_df.values
    T, _ = values.shape
    ar = np.full(T, np.nan)
    for t in range(window - 1, T):
        W = values[t - window + 1: t + 1]
        try:
            if estimator == "sample":
                Wc = W - W.mean(axis=0)
                Sigma = (Wc.T @ Wc) / (window - 1)
            elif estimator == "ledoit_wolf":
                Sigma = LedoitWolf().fit(W).covariance_
            elif estimator == "mcd":
                Sigma = MinCovDet(random_state=C.SEED).fit(W).covariance_
            elif estimator == "mp_filtered":
                Sigma = _mp_filter_covariance(W)
            else:
                raise ValueError(f"unknown estimator {estimator!r}")
            eig = np.maximum(np.linalg.eigvalsh(Sigma), 0.0)
            if eig.sum() > 0:
                ar[t] = eig[-1] / eig.sum()
        except Exception:
            ar[t] = np.nan
    return pd.Series(ar, index=returns_df.index, name=f"AR_{estimator}").dropna()


def compare_estimators(R_f: pd.DataFrame, window: int = C.WINDOW) -> pd.DataFrame:
    """Side-by-side AR_t across estimators on common dates."""
    series = {name: absorption_ratio_estimator(R_f, window, name)
              for name in ("sample", "ledoit_wolf", "mcd", "mp_filtered")}
    common = series["sample"].index
    for s in series.values():
        common = common.intersection(s.index)
    return pd.DataFrame({k: v.loc[common] for k, v in series.items()})


# ---------------------------------------------------------------------------
# Lead-lag stale-pricing diagnostic
# ---------------------------------------------------------------------------
def lead_lag_corr(x: pd.Series, y: pd.Series, max_lag: int = 5,
                  hac_lags: int = 5) -> pd.DataFrame:
    """corr(x_t, y_{t-k}) for k in [-max_lag, max_lag] with Newey-West p-values.

    k > 0 means y leads x (where a non-synchronous-close artifact would live).
    """
    import statsmodels.api as sm
    base = pd.concat([x, y], axis=1).dropna()
    base.columns = ["x", "y"]
    rows = []
    for k in range(-max_lag, max_lag + 1):
        d = pd.DataFrame({"x": base["x"], "y": base["y"].shift(k)}).dropna()
        r = d["x"].corr(d["y"])
        X = sm.add_constant(d["y"])
        res = sm.OLS(d["x"], X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        rows.append({"lag_k": k, "corr": r,
                     "p_HAC": res.pvalues.iloc[1], "n": len(d)})
    return pd.DataFrame(rows)
