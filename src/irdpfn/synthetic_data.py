"""
Synthetic data generator.

The real Lithuanian second-pillar NAV panel cannot be redistributed, so this
module fabricates a panel with the SAME statistical fingerprint used by the
paper, allowing the full pipeline to run end-to-end from a public repository:

  * 5 providers x 8 age cohorts = 40 funds, business days 2019-2025;
  * cohort volatility ladder (low for AG1 / AG8-TIPF, rising through AG2-AG7);
  * fat tails and negative skew (Student-t innovations + crisis down-jumps);
  * a common market factor whose loading SPIKES inside crisis windows, so the
    cross-section synchronises and the absorption ratio rises into a detectable
    High-Concentration regime;
  * a provider-level factor so DTW clustering recovers provider x segment groups;
  * a fund-specific benchmark index (small tracking error);
  * synthetic global equity / yield factors that share the crisis structure,
    so the augmented absorption ratio and lead-lag checks are reproducible
    fully offline (no yfinance / ECB calls).

Nothing here is calibrated to any real fund. It exists only so reviewers can
execute the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# Cohort base daily volatility (matches the published descriptive ladder)
_COHORT_VOL = {
    "AG1": 0.0022, "AG2": 0.0043, "AG3": 0.0069, "AG4": 0.0080,
    "AG5": 0.0081, "AG6": 0.0081, "AG7": 0.0083, "AG8": 0.0019,
}
# Loading on the common market factor (conservative cohorts load less)
_COHORT_BETA = {
    "AG1": 0.55, "AG2": 0.78, "AG3": 0.88, "AG4": 0.92,
    "AG5": 0.93, "AG6": 0.93, "AG7": 0.94, "AG8": 0.50,
}
_COHORT_DRIFT = {  # tiny positive daily mean
    "AG1": 1e-4, "AG2": 2e-4, "AG3": 4e-4, "AG4": 4e-4,
    "AG5": 4e-4, "AG6": 4e-4, "AG7": 4e-4, "AG8": 1e-4,
}


def _student_t(rng: np.random.Generator, df: float, size) -> np.ndarray:
    """Standardised Student-t innovations (unit variance for df > 2)."""
    raw = rng.standard_t(df, size=size)
    return raw / np.sqrt(df / (df - 2.0))


def _crisis_intensity(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Smooth 0..1 stress signal: 1 inside a published crisis window, with a
    short exponential ramp-down afterwards. Drives both the factor loading and
    the down-jump probability.
    """
    intensity = np.zeros(len(dates))
    for _name, (start, end, _c, _a) in C.EVENTS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        inside = (dates >= s) & (dates <= e)
        intensity[inside] = np.maximum(intensity[inside], 1.0)
        # 20-business-day decay after the window
        tail = (dates > e) & (dates <= e + pd.Timedelta(days=40))
        if tail.any():
            decay = np.exp(-np.arange(1, tail.sum() + 1) / 8.0)
            intensity[tail] = np.maximum(intensity[tail], decay)
    return intensity


def generate_panel(seed: int = C.SEED,
                   start: str = "2019-01-07",
                   end: str = "2025-09-29") -> pd.DataFrame:
    """
    Build the long-format synthetic panel.

    Returns
    -------
    DataFrame with columns
        Date, AgeGroup, Provider, log_return_price, log_return_index
    where `Provider` holds the fund id ("Provider 1_AG1") and `AgeGroup`
    holds the age band ("54/60"), matching the real export schema.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)

    stress = _crisis_intensity(dates)

    # --- common market factor: vol and a negative drift amplify in crisis ----
    base_factor_vol = 1.0
    factor_vol = base_factor_vol * (1.0 + 3.0 * stress)
    factor = _student_t(rng, df=4.0, size=T) * factor_vol
    factor -= stress * 1.6                      # crisis sell-off -> negative skew
    factor -= rng.binomial(1, 0.03 + 0.25 * stress, T) * \
        np.abs(_student_t(rng, df=3.0, size=T)) * (0.7 + 1.6 * stress)
    factor = (factor - factor.mean()) / factor.std()

    # --- provider factors: give within-provider co-movement (clustering) -----
    prov_factor = {
        p: _student_t(rng, df=4.0, size=T) for p in C.PROVIDERS
    }

    # --- factor weight w_t: average corr ~ w^2 (calm ~0.85, crisis ~0.97) ----
    w_t = 0.925 + 0.06 * stress
    w_t = np.clip(w_t, 0.0, 0.99)

    rows = []
    for p_idx, provider in enumerate(C.PROVIDERS):
        # mild provider personality on loadings and vol
        prov_beta_mult = 0.92 + 0.04 * p_idx
        prov_vol_mult = 0.95 + 0.025 * p_idx
        g = prov_factor[provider]

        for cohort in C.COHORTS:
            sigma = _COHORT_VOL[cohort] * prov_vol_mult
            beta = _COHORT_BETA[cohort] * prov_beta_mult
            beta = min(beta, 0.97)

            eps = _student_t(rng, df=6.0, size=T)
            # systematic part: global + provider, normalised to unit variance
            sys = beta * factor + 0.26 * g
            sys = sys / sys.std()
            # blend systematic vs idiosyncratic by the time-varying weight
            unit = w_t * sys + np.sqrt(1.0 - w_t ** 2) * eps
            r_price = _COHORT_DRIFT[cohort] + sigma * unit

            # fund-specific benchmark: fund minus small active return
            active = rng.normal(0.0, 0.20 * sigma, T)
            r_index = r_price - active

            band = C.COHORT_BANDS[cohort]
            fund_id = f"{provider}_{cohort}"
            rows.append(pd.DataFrame({
                "Date": dates,
                "AgeGroup": band,
                "Provider": fund_id,
                "log_return_price": r_price,
                "log_return_index": r_index,
            }))

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["Provider", "Date"]).reset_index(drop=True)
    return panel


def generate_global_factors(seed: int = C.SEED,
                            start: str = "2019-01-07",
                            end: str = "2025-09-29") -> pd.DataFrame:
    """
    Synthetic global benchmark matrix G^(b) = [3 equity returns | 2 yield
    shocks(bp)], date-indexed. Equity co-moves moderately with crisis stress;
    yields are a separate fat-tailed bp-scale process.
    """
    rng = np.random.default_rng(seed + 1)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    stress = _crisis_intensity(dates)

    eq_common = _student_t(rng, df=4.0, size=T) * (1.0 + 3.0 * stress)
    eq_common -= stress * 2.0
    eq_common = (eq_common - eq_common.mean()) / eq_common.std()

    equity = {}
    for name, vol in zip(C.EQUITY_COLS, (0.0128, 0.0121, 0.0127)):
        idio = _student_t(rng, df=5.0, size=T)
        e = 0.85 * eq_common + np.sqrt(1 - 0.85 ** 2) * idio
        equity[name] = 5e-4 + vol * e

    yields = {}
    for name, scale in zip(C.YIELD_COLS, (4.8, 4.55)):
        y = _student_t(rng, df=4.0, size=T) * scale * (1.0 + 1.5 * stress)
        yields[name] = 0.15 + y

    g = pd.DataFrame({**equity, **yields}, index=dates)
    g.index.name = "Date"
    return g


def write_synthetic_csv(path=C.SYNTHETIC_CSV, seed: int = C.SEED) -> pd.DataFrame:
    """Generate the fund panel and persist it as the public data file."""
    panel = generate_panel(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(path, index=False)
    return panel
