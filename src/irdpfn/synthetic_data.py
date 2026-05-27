"""
Synthetic pension fund panel generator.

Produces a realistic toy dataset with the same structure as the real Bank
of Lithuania panel: 5 providers x 8 age cohorts = 40 funds, daily log
returns over 2019-2025, with embedded regime structure so that the
downstream pipeline produces meaningful figures.

Key design choices
------------------
- Three regimes (Low / Moderate / High) driven by a hidden Markov chain
  with sticky transitions, mimicking the empirical absorption ratio.
- Regime-dependent factor loadings on a single common factor produce
  the synchronisation pattern picked up by the absorption ratio.
- Age cohort heterogeneity: AG1 (conservative) and AG8 (aggressive)
  cohorts have idiosyncratic risk profiles; middle cohorts cluster.
- Fund-specific benchmarks track the funds with regime-dependent
  tracking error.

This is NOT a substitute for the real data, but it lets you run
the pipeline end-to-end and reproduce qualitatively similar figures.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    AGE_GROUPS,
    DATA_DIR,
    N_COHORTS,
    N_PROVIDERS,
    PROVIDERS,
    SEED,
)


def _regime_sequence(T, rng):
    """Generate hidden regime path (0=Low, 1=Moderate, 2=High)."""
    # Sticky transition matrix loosely calibrated to the real data:
    # ~ 1-year average persistence in Low/Mod, ~6 months in High.
    P = np.array([
        [0.992, 0.007, 0.001],
        [0.005, 0.990, 0.005],
        [0.001, 0.011, 0.988],
    ])

    states = np.zeros(T, dtype=int)
    states[0] = 0
    for t in range(1, T):
        states[t] = rng.choice(3, p=P[states[t-1]])
    return states


def _factor_loadings(rng):
    """
    Build the factor-loading matrix B[N_funds].

    AG1 (conservative): low loadings.
    AG8 (TIPF, aggressive): high loadings.
    Middle cohorts: monotonic interpolation with provider noise.
    """
    base = np.linspace(0.4, 1.1, N_COHORTS)   # AG1 -> AG8
    B = np.zeros((N_PROVIDERS, N_COHORTS))
    for p in range(N_PROVIDERS):
        provider_shift = rng.normal(0, 0.05)
        B[p, :] = base + provider_shift + rng.normal(0, 0.02, N_COHORTS)
    return np.clip(B, 0.2, 1.5)


def generate_panel(
    start="2019-01-01",
    end="2025-12-31",
    seed=SEED,
    out_path=None,
):
    """
    Generate the synthetic panel and write it to disk as CSV.

    Returns
    -------
    df       : long-format DataFrame
    out_path : path to the written file
    """
    rng = np.random.default_rng(seed)

    # Trading days (Mon-Fri)
    dates = pd.bdate_range(start=start, end=end)
    T = len(dates)

    # Regime path drives volatility AND cross-fund correlation
    states = _regime_sequence(T, rng)

    # Regime-specific parameters
    factor_vol_by_regime = np.array([0.004, 0.007, 0.020])   # Low/Mod/High
    idio_vol_by_regime   = np.array([0.003, 0.004, 0.006])
    bench_vol_by_regime  = np.array([0.003, 0.005, 0.012])

    # Factor returns (one global factor)
    factor = factor_vol_by_regime[states] * rng.standard_normal(T)

    # Fund loadings
    B = _factor_loadings(rng)   # shape (N_PROVIDERS, N_COHORTS)

    rows = []
    for p_idx, provider in enumerate(PROVIDERS):
        for ag_idx, ag in enumerate(AGE_GROUPS):
            b = B[p_idx, ag_idx]
            idio = idio_vol_by_regime[states] * rng.standard_normal(T)
            r_fund = b * factor + idio

            bench_noise = bench_vol_by_regime[states] * rng.standard_normal(T)
            r_bench = b * factor * 0.95 + 0.4 * idio + bench_noise

            for t, date in enumerate(dates):
                rows.append({
                    "Date":             date,
                    "Provider":         provider,
                    "AgeGroup":         ag,
                    "log_return_price": r_fund[t],
                    "log_return_index": r_bench[t],
                })

    df = pd.DataFrame(rows)
    out_path = Path(out_path) if out_path else DATA_DIR / "pension_fund_synthetic.csv"
    df.to_csv(out_path, index=False)

    return df, out_path


if __name__ == "__main__":
    df, path = generate_panel()
    print(f"Wrote {len(df):,} rows to {path}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Funds: {df['Provider'].nunique()} providers "
          f"x {df['AgeGroup'].nunique()} cohorts "
          f"= {df['Provider'].nunique() * df['AgeGroup'].nunique()}")
