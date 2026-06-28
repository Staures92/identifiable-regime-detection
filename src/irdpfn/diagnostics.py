"""
Interpretive and robustness diagnostics for AR_t.

Correlations of AR_t with the interpretive series (Pearson / Spearman /
Kendall), a regime-conditional OLS with Newey-West standard errors, the
baseline-vs-augmented AR agreement, and a permutation test for the
crisis-vs-calm co-movement gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats

from . import config as C


def _sig(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))


# ---------------------------------------------------------------------------
# Interpretive frame
# ---------------------------------------------------------------------------
def build_interp_frame(AR, R_f, R_bf, G_b, regime_series=None) -> pd.DataFrame:
    """AR_t aligned with MAAR and the global factors (+ optional regime)."""
    maar = (R_f - R_bf).abs().mean(axis=1).rename("MAAR")
    g = G_b.reindex(AR.index)
    df = pd.concat([AR.rename("AR"), maar.reindex(AR.index), g], axis=1)
    df.columns = ["AR", "MAAR"] + list(g.columns)
    if regime_series is not None:
        df["Regime"] = regime_series.reindex(df.index)
    return df.dropna()


def correlation_table(interp_df: pd.DataFrame,
                      series_cols=("MAAR", *C.GLOBAL_COLS)) -> pd.DataFrame:
    """Pearson + Spearman + Kendall correlations of each series with AR_t."""
    rows = []
    for col in series_cols:
        x, y = interp_df["AR"], interp_df[col]
        r, pr = sstats.pearsonr(x, y)
        rho, ps = sstats.spearmanr(x, y)
        tau, pk = sstats.kendalltau(x, y)
        rows.append({"Series": col,
                     "Pearson_r": round(r, 4), "Pearson_sig": _sig(pr),
                     "Spearman_rho": round(rho, 4), "Spearman_sig": _sig(ps),
                     "Kendall_tau": round(tau, 4), "Kendall_sig": _sig(pk)})
    return pd.DataFrame(rows)


def regime_conditional_ols(interp_df: pd.DataFrame,
                           series_cols=("MAAR", *C.GLOBAL_COLS)):
    """y = b0 + b1*AR + b2*D_High + b3*(AR*D_High); HAC (Newey-West) SEs.

    Significant b3 => the AR relationship is regime-dependent (not continuous).
    """
    import statsmodels.api as sm
    T = len(interp_df)
    D_high = (interp_df["Regime"] == "High-Concentration").astype(int).values
    hac = int(np.floor(4 * (T / 100) ** (2 / 9)))
    fitted, rows = {}, []
    for col in series_cols:
        y = interp_df[col].values
        X = sm.add_constant(np.column_stack([
            interp_df["AR"].values, D_high, interp_df["AR"].values * D_high]))
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
        fitted[col] = res
        rows.append({"Series": col,
                     "b1": round(res.params[1], 4), "b1_p": round(res.pvalues[1], 4),
                     "b3": round(res.params[3], 4), "b3_p": round(res.pvalues[3], 4),
                     "R2": round(res.rsquared, 4)})
    return pd.DataFrame(rows), fitted


# ---------------------------------------------------------------------------
# Augmented-AR agreement + permutation test
# ---------------------------------------------------------------------------
def augmented_agreement(AR_corr, AR_augmented, events=C.EVENTS):
    """Overall / crisis / calm correlation between baseline and augmented AR."""
    idx = AR_corr.index.intersection(AR_augmented.index)
    b, a = AR_corr.loc[idx].values, AR_augmented.loc[idx].values
    rho = np.corrcoef(b, a)[0, 1]
    mask = pd.Series(False, index=idx)
    for _n, (s, e, _c, _al) in events.items():
        mask |= (idx >= pd.Timestamp(s)) & (idx <= pd.Timestamp(e))
    mask = mask.values
    rho_crisis = np.corrcoef(b[mask], a[mask])[0, 1]
    rho_calm = np.corrcoef(b[~mask], a[~mask])[0, 1]
    return {"idx": idx, "b": b, "a": a, "mask": mask, "rho_overall": rho,
            "rho_crisis": rho_crisis, "rho_calm": rho_calm,
            "gap": rho_crisis - rho_calm, "n_crisis": int(mask.sum())}


def permutation_gap_test(agree: dict, n_perm: int = 2000, seed: int = 42):
    """Is the crisis-calm co-movement gap larger than random splits give?"""
    rng = np.random.default_rng(seed)
    b, a, mask = agree["b"], agree["a"], agree["mask"]
    obs = agree["gap"]
    n_c, n = int(mask.sum()), len(b)
    null = np.empty(n_perm)
    for i in range(n_perm):
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, n_c, replace=False)] = True
        null[i] = np.corrcoef(b[m], a[m])[0, 1] - np.corrcoef(b[~m], a[~m])[0, 1]
    p = (np.abs(null) >= abs(obs)).mean()
    return {"observed_gap": obs, "p_perm": float(p)}


# ---------------------------------------------------------------------------
# Cross-step: cluster MAAR amplification, normal vs crisis
# ---------------------------------------------------------------------------
def cluster_maar_amplification(R_f, R_bf, cluster_df, regime_series, ar_index):
    """Mean absolute active return per cluster in crisis vs normal regimes."""
    maar_daily = (R_f - R_bf).abs().reindex(ar_index)
    crisis = ar_index[regime_series == "High-Concentration"]
    normal = ar_index[regime_series != "High-Concentration"]
    rows = []
    for cid in sorted(cluster_df["Cluster"].unique()):
        funds = cluster_df[cluster_df["Cluster"] == cid]["Fund"].tolist()
        m = maar_daily[funds].mean(axis=1)
        mn, mc = m.loc[normal].mean(), m.loc[crisis].mean()
        ratio = mc / mn if (pd.notna(mn) and mn > 0) else np.nan
        cohorts = set(cluster_df[cluster_df["Cluster"] == cid]["AgeCohort"])
        ctype = ("Extreme (AG1/AG8)"
                 if cohorts.issubset(C.CONSERVATIVE_COHORTS) else "Middle (AG2-AG7)")
        rows.append({"Cluster": f"C{cid}", "N_funds": len(funds),
                     "MAAR_normal": mn, "MAAR_crisis": mc,
                     "Ratio": ratio, "Cohort_type": ctype})
    return pd.DataFrame(rows).set_index("Cluster")
