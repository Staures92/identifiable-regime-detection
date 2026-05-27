"""
Project-wide configuration.

All paths, hyperparameters, and styling defaults live here so that
scripts and modules share a single source of truth.
"""

from pathlib import Path

# =========================================================
# PATHS
# =========================================================
ROOT_DIR     = Path(__file__).resolve().parents[2]
DATA_DIR     = ROOT_DIR / "data"
FIGURES_DIR  = ROOT_DIR / "figures"
OUTPUTS_DIR  = ROOT_DIR / "outputs"

for _p in (DATA_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Default data file. Replace with `pension_fund_real.csv` for the
# real Bank of Lithuania extract; the synthetic generator writes
# `pension_fund_synthetic.csv`.
DEFAULT_DATA_FILE = DATA_DIR / "pension_fund_synthetic.csv"


# =========================================================
# DATA STRUCTURE
# =========================================================
N_FUNDS     = 40         # 5 providers x 8 age cohorts
N_PROVIDERS = 5
N_COHORTS   = 8
PROVIDERS   = [f"Provider {i}" for i in range(1, N_PROVIDERS + 1)]
AGE_GROUPS  = [f"AG{i}"        for i in range(1, N_COHORTS   + 1)]

# Global benchmarks (Yahoo Finance tickers)
GLOBAL_TICKERS = {
    "MSCI_World":  "URTH",
    "MSCI_Europe": "IEUR",
    "SP500":       "^GSPC",
}


# =========================================================
# ABSORPTION RATIO
# =========================================================
ROLLING_WINDOW = 60      # trading days


# =========================================================
# CLUSTERING
# =========================================================
K_CLUSTER_RANGE = range(2, 16)
K_CLUSTER_STAR  = 8      # optimal K_c* by Davies-Bouldin


# =========================================================
# HMM REGIME DETECTION
# =========================================================
K_REGIME_RANGE_BIC  = range(2, 9)
K_REGIME_RANGE_NUTS = [3, 4, 5, 6]
K_REGIME_BASELINE   = 3
KAPPA_SENSITIVITY   = [10.0, 20.0, 30.0, 50.0]
SEED                = 42

REGIME_NAMES = ["Low-Risk", "Moderate-Risk", "High-Risk"]


# =========================================================
# VISUAL STYLE
# =========================================================
PROVIDER_COLORS = {
    "Provider 1": "#1f77b4",
    "Provider 2": "#ff7f0e",
    "Provider 3": "#2ca02c",
    "Provider 4": "#d62728",
    "Provider 5": "#9467bd",
}

REGIME_COLORS = {
    "Low-Risk":      "#2166ac",   # blue
    "Moderate-Risk": "#fdae61",   # amber
    "High-Risk":     "#d73027",   # red
}

GLOBAL_BENCHMARK_STYLES = {
    "MSCI_World":  ("#1f77b4", "-",  1.5, "MSCI World"),
    "MSCI_Europe": ("#ff7f0e", "--", 1.5, "MSCI Europe"),
    "SP500":       ("#d62728", "-.", 1.5, "S&P 500"),
}

PLT_RCPARAMS = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        300,
}


# =========================================================
# KEY EVENT WINDOWS (for figure shading)
# =========================================================
EVENT_WINDOWS = {
    "COVID-19":        ("2020-02-19", "2020-05-23", "#d73027", 0.15),
    "2022 rate hikes": ("2022-03-01", "2022-12-31", "#ff7f0e", 0.15),
}
