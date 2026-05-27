"""
Absorption ratio and rolling-window risk decomposition.

Definition
----------
AR_t = lambda_1(t) / trace(Sigma_t)

where Sigma_t is the sample covariance of returns over the rolling
window [t - omega + 1, t]. lambda_1(t) is its largest eigenvalue.

Risk decomposition:
    Total      = trace(Sigma_t)
    Systematic = lambda_1(t)
    Specific   = Total - Systematic
    AR_t       = Systematic / Total
"""

import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf, MinCovDet
from .config import ROLLING_WINDOW


# =========================================================
# 1. BASELINE ABSORPTION RATIO
# =========================================================
def compute_absorption_ratio(returns_df, window=ROLLING_WINDOW):
    """
    Rolling-window absorption ratio.

    Regularisation (+1e-6 * I) is applied ONLY when N > window, i.e.
    when the covariance matrix is rank-deficient. Applying it to the
    full-rank baseline materially distorts AR_t.
    """
    values = returns_df.values
    T, N   = values.shape
    ar     = np.full(T, np.nan)

    for t in range(window - 1, T):
        W        = values[t - window + 1 : t + 1]
        centered = W - W.mean(axis=0)
        Sigma    = (centered.T @ centered) / (window - 1)

        if N > window:
            Sigma += 1e-6 * np.eye(N)

        try:
            eigvals = np.linalg.eigvalsh(Sigma)
            eigvals = np.maximum(eigvals, 0)
            total   = eigvals.sum()
            if total > 0:
                ar[t] = eigvals[-1] / total
        except np.linalg.LinAlgError:
            ar[t] = np.nan

    return pd.Series(ar, index=returns_df.index, name="AR")


# =========================================================
# 2. RISK DECOMPOSITION
# =========================================================
def compute_risk_decomposition(returns_df, window=ROLLING_WINDOW):
    """
    Rolling-window decomposition into Total / Systematic / Specific risk.
    """
    values  = returns_df.values
    T, _    = values.shape
    records = []

    for t in range(window - 1, T):
        W        = values[t - window + 1 : t + 1]
        centered = W - W.mean(axis=0)
        Sigma    = (centered.T @ centered) / (window - 1)

        try:
            eigvals    = np.linalg.eigvalsh(Sigma)
            eigvals    = np.maximum(eigvals, 0)
            total      = eigvals.sum()
            systematic = eigvals[-1]
            specific   = total - systematic
            if total > 0:
                records.append({
                    "Date":            returns_df.index[t],
                    "Total_Risk":      total,
                    "Systematic_Risk": systematic,
                    "Specific_Risk":   specific,
                    "AR_t":            systematic / total,
                    "Specific_Share":  specific / total,
                })
        except np.linalg.LinAlgError:
            continue

    return pd.DataFrame(records).set_index("Date")


# =========================================================
# 3. ALTERNATIVE COVARIANCE ESTIMATORS (robustness)
# =========================================================
def compute_ar_with_estimator(returns_df, window=ROLLING_WINDOW,
                              estimator="sample"):
    """
    Compute AR_t under alternative covariance estimators.

    estimator : {"sample", "ledoit_wolf", "mcd"}
    """
    values = returns_df.values
    T, _   = values.shape
    ar     = np.full(T, np.nan)

    for t in range(window - 1, T):
        W = values[t - window + 1 : t + 1]
        try:
            if estimator == "sample":
                centered = W - W.mean(axis=0)
                Sigma    = (centered.T @ centered) / (window - 1)
            elif estimator == "ledoit_wolf":
                Sigma = LedoitWolf().fit(W).covariance_
            elif estimator == "mcd":
                Sigma = MinCovDet(random_state=42).fit(W).covariance_
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

            eigvals = np.linalg.eigvalsh(Sigma)
            eigvals = np.maximum(eigvals, 0)
            total   = eigvals.sum()
            if total > 0:
                ar[t] = eigvals[-1] / total
        except Exception:
            ar[t] = np.nan

    return pd.Series(
        ar, index=returns_df.index, name=f"AR_{estimator}",
    ).dropna()


# =========================================================
# 4. REGULARISATION VERIFICATION (footnote 1)
# =========================================================
def verify_regularisation(R_f, window=ROLLING_WINDOW):
    """
    Confirm that applying epsilon * I to a full-rank baseline does NOT
    materially affect AR_t. Reported in footnote 1.
    """
    results = []
    n_total = len(R_f)
    samples = [
        ("Early window", 0),
        ("Mid window",   n_total // 2),
        ("Late window",  n_total - window),
    ]
    N = R_f.shape[1]

    for label, start in samples:
        W        = R_f.values[start : start + window]
        centered = W - W.mean(axis=0)
        Sigma    = (centered.T @ centered) / (window - 1)
        ev_u     = np.linalg.eigvalsh(Sigma)
        ev_r     = np.linalg.eigvalsh(Sigma + 1e-6 * np.eye(N))
        results.append({
            "Window":     label,
            "AR_unreg":   round(ev_u[-1] / ev_u.sum(), 6),
            "AR_reg":     round(ev_r[-1] / ev_r.sum(), 6),
            "Difference": round(abs(ev_u[-1] / ev_u.sum()
                                    - ev_r[-1] / ev_r.sum()), 6),
        })
    return pd.DataFrame(results)


# =========================================================
# 5. TRACKING ERROR
# =========================================================
def tracking_error(R_f, R_bf, kind="mae"):
    """
    Daily cross-fund tracking error.

    kind : "mae" -> mean absolute deviation across funds (default)
           "std" -> cross-sectional std of deviations
    """
    diff = (R_f - R_bf).abs()
    if kind == "mae":
        te = diff.mean(axis=1)
    elif kind == "std":
        te = (R_f - R_bf).std(axis=1)
    else:
        raise ValueError(f"Unknown kind: {kind}")
    te.name = "Tracking_Error"
    return te
