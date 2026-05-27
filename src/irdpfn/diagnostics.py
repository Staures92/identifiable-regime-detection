"""
Diagnostic tests for the AR-benchmark relationship.

- Pearson / Spearman / Kendall correlations with significance.
- Regime-conditional OLS regression with HAC (Newey-West) SE.
- Cluster x regime tracking error amplification.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import statsmodels.api as sm


# =========================================================
# 1. CORRELATION SIGNIFICANCE (3 measures)
# =========================================================
def _sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def correlation_tests(interp_df, target_col="AR",
                      series_cols=None):
    """
    Compute Pearson, Spearman, Kendall correlations between `target_col`
    and each of `series_cols`, with significance markers.
    """
    if series_cols is None:
        series_cols = [c for c in interp_df.columns
                       if c != target_col and c != "Regime"]

    T = len(interp_df)
    rows = []
    for col in series_cols:
        x = interp_df[target_col].values
        y = interp_df[col].values

        r, p_p = scipy_stats.pearsonr(x, y)
        t_p    = r * np.sqrt((T - 2) / (1 - r**2)) if abs(r) < 1 else np.inf

        rho, p_s = scipy_stats.spearmanr(x, y)
        tau, p_k = scipy_stats.kendalltau(x, y)

        rows.append({
            "Series":       col,
            "Pearson_r":    round(r, 4),
            "t_statistic":  round(t_p, 4),
            "Pearson_p":    round(p_p, 6),
            "Pearson_sig":  _sig_marker(p_p),
            "Spearman_rho": round(rho, 4),
            "Spearman_p":   round(p_s, 6),
            "Spearman_sig": _sig_marker(p_s),
            "Kendall_tau":  round(tau, 4),
            "Kendall_p":    round(p_k, 6),
            "Kendall_sig":  _sig_marker(p_k),
        })
    return pd.DataFrame(rows)


# =========================================================
# 2. REGIME-CONDITIONAL OLS WITH HAC SE
# =========================================================
def regime_conditional_regression(interp_df, series_cols,
                                  ar_col="AR", regime_col="Regime",
                                  high_label="High-Risk"):
    """
    Fit y_t = b0 + b1*AR + b2*D_high + b3*(AR*D_high) + e_t for each
    series, with Newey-West HAC standard errors.

    Returns
    -------
    fitted : dict[str, RegressionResults]
    table  : DataFrame with key coefficients per series
    """
    T = len(interp_df)
    D_high = (interp_df[regime_col] == high_label).astype(int).values
    hac_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

    fitted, rows = {}, []
    for col in series_cols:
        y = interp_df[col].values
        X = np.column_stack([
            interp_df[ar_col].values,
            D_high,
            interp_df[ar_col].values * D_high,
        ])
        X = sm.add_constant(X)
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
        fitted[col] = res

        rows.append({
            "Series":  col,
            "b0":      round(res.params[0], 4),
            "b1":      round(res.params[1], 4),
            "b1_se":   round(res.bse[1],    4),
            "b1_p":    round(res.pvalues[1],4),
            "b2":      round(res.params[2], 4),
            "b2_se":   round(res.bse[2],    4),
            "b2_p":    round(res.pvalues[2],4),
            "b3":      round(res.params[3], 4),
            "b3_se":   round(res.bse[3],    4),
            "b3_p":    round(res.pvalues[3],4),
            "R2":      round(res.rsquared,  4),
            "N":       int(res.nobs),
            "HAC_lags": hac_lags,
        })
    return fitted, pd.DataFrame(rows)


# =========================================================
# 3. CLUSTER × REGIME TRACKING ERROR AMPLIFICATION
# =========================================================
def cluster_regime_tracking_error(cluster_df, R_f, R_bf,
                                  AR_clean, regime_series,
                                  high_label="High-Risk"):
    """
    For each cluster, compute mean tracking error in normal regimes vs
    in the High-Risk regime, plus the crisis/normal amplification ratio.

    Parameters
    ----------
    cluster_df : DataFrame with columns ['Fund', 'Cluster']
    """
    te_daily = (R_f - R_bf).abs()
    te_align = te_daily.reindex(AR_clean.index)

    crisis_dates = AR_clean.index[regime_series == high_label]
    normal_dates = AR_clean.index[regime_series != high_label]

    rows = []
    for c in sorted(cluster_df["Cluster"].unique()):
        funds = cluster_df[cluster_df["Cluster"] == c]["Fund"].tolist()
        te_c  = te_align[funds].mean(axis=1)

        te_n = te_c.loc[normal_dates].mean()
        te_x = te_c.loc[crisis_dates].mean()
        ratio = te_x / te_n if te_n > 0 else np.nan

        ag_labels = sorted({f.split("_")[1] for f in funds})
        cohort_type = ("Extreme (AG1+AG8)"
                       if all(ag in ["AG1", "AG8"] for ag in ag_labels)
                       else "Middle (AG2-AG7)")

        rows.append({
            "Cluster":      f"C{c}",
            "N_funds":      len(funds),
            "TE_normal":    te_n,
            "TE_crisis":    te_x,
            "Ratio":        ratio,
            "Cohort_type":  cohort_type,
            "Funds":        funds,
        })
    return pd.DataFrame(rows).set_index("Cluster")


# =========================================================
# 4. BUILD UNIFIED INTERPRETATION DATAFRAME
# =========================================================
def build_interp_df(AR_baseline, tracking_err_series, R_bg,
                    regime_series=None):
    """
    Combine AR_t, tracking error, global benchmarks, and (optionally)
    HMM regime labels into a single aligned DataFrame.
    """
    R_bg_aligned = R_bg.reindex(AR_baseline.index)

    parts = [AR_baseline,
             tracking_err_series.reindex(AR_baseline.index),
             R_bg_aligned]
    cols  = ["AR", "Tracking_Error"] + list(R_bg.columns)

    if regime_series is not None:
        parts.append(regime_series.reindex(AR_baseline.index))
        cols.append("Regime")

    df = pd.concat(parts, axis=1)
    df.columns = cols
    return df.dropna()
