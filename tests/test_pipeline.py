"""
Smoke and property tests for the irdpfn pipeline.

These run on a small synthetic panel and check structural invariants rather than
exact numbers (which are data-dependent). They are deliberately cheap: the EM
fits use few restarts and the Bayesian sweep is never invoked here.

    pytest -q
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from irdpfn import (synthetic_data, data_io, absorption as ab, clustering as cl,
                    regime as rg, diagnostics as dg, config as C)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    synthetic_data.write_synthetic_csv(path, seed=C.SEED)
    return data_io.load_panel(path)


@pytest.fixture(scope="module")
def ar_cov(panel):
    return ab.absorption_ratio(panel.R_f, window=C.WINDOW,
                               method="covariance").dropna()


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def test_panel_shapes(panel):
    assert panel.R_f.shape[1] == C.N
    assert panel.G_b.shape[1] == C.M
    # augmented matrix is [R_f | R_bf | G_b] = 2N + M
    assert panel.R_aug.shape[1] == 2 * C.N + C.M
    assert len(panel.R_f) == len(panel.R_bf) == len(panel.G_b)


def test_cohort_volatility_ladder(panel):
    # younger cohorts (AG2..AG7) should be more volatile than the AG1/TIPF floors
    std = panel.R_f.std()
    ag1 = std[[c for c in panel.R_f.columns if c.endswith("AG1")]].mean()
    ag7 = std[[c for c in panel.R_f.columns if c.endswith("AG7")]].mean()
    assert ag7 > ag1


def test_returns_fat_tailed(panel):
    # pooled excess kurtosis well above the Gaussian value of 0
    flat = panel.R_f.values.ravel()
    flat = flat[np.isfinite(flat)]
    k = pd.Series(flat).kurtosis()
    assert k > 3.0


# ---------------------------------------------------------------------------
# Absorption ratio
# ---------------------------------------------------------------------------
def test_ar_bounded(ar_cov):
    assert ar_cov.between(0.0, 1.0).all()


def test_correlation_ar_exact_mapping(panel):
    # AR_corr maps to mean off-diagonal correlation via rho = (N*AR - 1)/(N-1)
    ar = ab.absorption_ratio(panel.R_f, window=C.WINDOW, method="correlation").dropna()
    rho = ab.rho_from_ar(ar.values, C.N)
    assert np.all(rho <= 1.0 + 1e-9)
    assert np.all(rho >= -1.0 / (C.N - 1) - 1e-6)


def test_augmented_below_baseline(panel):
    ar_corr = ab.absorption_ratio(panel.R_f, window=C.WINDOW,
                                  method="correlation").dropna()
    ar_aug = ab.absorption_ratio(panel.R_aug, window=C.WINDOW,
                                 method="correlation").dropna()
    assert ar_aug.mean() < ar_corr.mean()


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def test_clustering_runs(panel):
    res = cl.run_clustering(panel.R_f, panel.R_bf,
                            robustness=False, noncircularity=True)
    assert 2 <= res.best_k <= 15
    assert len(res.labels_hier) == C.N
    # non-circularity table should exist and recover provider structure strongly
    nc = res.noncircularity
    prov = nc.loc[nc["Reference partition"].str.startswith("Provider only"),
                  "Hierarchical"].iloc[0]
    assert prov > 0.5


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------
def test_regime_pipeline(ar_cov):
    res = rg.run_hmm_pipeline(ar_cov, label="test", n_seeds=5, rho_exact=False)
    mu = res["mu"]
    # three ordered, separated emission means
    assert len(mu) == C.K_BASELINE
    assert mu[0] < mu[1] < mu[2]
    # crisis threshold sits strictly between Moderate and High means
    assert mu[1] < res["tau"] < mu[2]
    # tau_1 sits between Low and Moderate
    assert mu[0] < res["tau_1"] < mu[1]


def test_gaussian_crossing_between_means():
    tau = rg.gaussian_crossing(0.66, 0.04, 0.93, 0.05)
    assert 0.66 < tau < 0.93


def test_kappa_from_persistence_monotone():
    # higher self-persistence implies a stickier prior (larger kappa)
    k_lo = rg.kappa_from_persistence(0.90, C.K_BASELINE)
    k_hi = rg.kappa_from_persistence(0.99, C.K_BASELINE)
    assert k_hi > k_lo


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def test_interp_frame_and_correlations(panel, ar_cov):
    res = rg.run_hmm_pipeline(ar_cov, label="test", n_seeds=5, rho_exact=False)
    interp = dg.build_interp_frame(ar_cov, panel.R_f, panel.R_bf, panel.G_b,
                                   res["regime_series"])
    assert {"AR", "MAAR", "Regime"}.issubset(interp.columns)
    corr = dg.correlation_table(interp)
    assert "Pearson_r" in corr.columns
    assert corr["Pearson_r"].abs().le(1.0).all()
