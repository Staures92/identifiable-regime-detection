"""
Smoke test for the CASRI pipeline.

Verifies that the core modules import and that the absorption ratio,
clustering, and EM-based HMM steps run on a small synthetic sample
without raising. Does NOT run the Bayesian sweep — that's covered in
the full pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="module")
def synthetic_returns():
    """Build a tiny return panel: 200 days x 10 funds."""
    rng = np.random.default_rng(0)
    T, N = 200, 10
    dates = pd.bdate_range("2023-01-01", periods=T)
    factor = rng.normal(0, 0.01, T)
    loadings = rng.uniform(0.5, 1.0, N)
    idio = rng.normal(0, 0.005, (T, N))
    returns = loadings[None, :] * factor[:, None] + idio
    cols = [f"Provider {i+1}_AG{(i % 8) + 1}" for i in range(N)]
    return pd.DataFrame(returns, index=dates, columns=cols)


def test_imports():
    """Every module imports cleanly."""
    from irdpfn import (
        absorption,
        clustering,
        config,
        data_io,
        diagnostics,
        figures,
        regime,
        synthetic_data,
    )


def test_absorption_ratio(synthetic_returns):
    from irdpfn.absorption import compute_absorption_ratio
    ar = compute_absorption_ratio(synthetic_returns, window=30).dropna()
    assert len(ar) > 0
    assert (ar.between(0, 1)).all()


def test_risk_decomposition(synthetic_returns):
    from irdpfn.absorption import compute_risk_decomposition
    rd = compute_risk_decomposition(synthetic_returns, window=30)
    assert {"Total_Risk", "Systematic_Risk",
            "Specific_Risk", "AR_t"} <= set(rd.columns)
    # AR_t == Systematic / Total
    np.testing.assert_allclose(
        rd["AR_t"],
        rd["Systematic_Risk"] / rd["Total_Risk"],
        atol=1e-10,
    )


def test_dtw_clustering(synthetic_returns):
    from irdpfn.clustering import compute_dtw_linkage, db_score_sweep_ward
    Z, scaled, names = compute_dtw_linkage(synthetic_returns)
    assert Z.shape == (synthetic_returns.shape[1] - 1, 4)
    db = db_score_sweep_ward(Z, scaled, K_range=range(2, 5))
    assert all(v > 0 for v in db.values())


def test_em_hmm(synthetic_returns):
    from irdpfn.absorption import compute_absorption_ratio
    from irdpfn.regime import fit_em_hmms, characterise_regimes, compute_thresholds
    ar = compute_absorption_ratio(synthetic_returns, window=30).dropna()
    ar_input = ar.values.reshape(-1, 1)
    models, bic, K_bic = fit_em_hmms(ar_input, K_range=range(2, 4),
                                     n_seeds=3, verbose=False)
    assert K_bic in {2, 3}
    # Need K=3 to test characterisation with default regime names
    if 3 in models:
        from irdpfn.config import REGIME_NAMES
        labels, summary, info = characterise_regimes(
            models[3], ar_input, ar.index,
        )
        assert len(labels) == len(ar)
        assert set(labels.unique()) <= set(REGIME_NAMES)
        thr = compute_thresholds(models[3], synthetic_returns.shape[1],
                                 ar)
        assert thr["tau"] > thr["tau_1"]


def test_synthetic_data_generator(tmp_path):
    from irdpfn.synthetic_data import generate_panel
    out = tmp_path / "test_panel.csv"
    df, path = generate_panel(start="2024-01-01", end="2024-06-30",
                              out_path=out)
    assert path == out
    assert path.exists()
    assert {"Date", "Provider", "AgeGroup",
            "log_return_price", "log_return_index"} <= set(df.columns)
    assert df["Provider"].nunique() == 5
    assert df["AgeGroup"].nunique() == 8
