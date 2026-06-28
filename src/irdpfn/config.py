"""
Central configuration for the IRDPFN pipeline.

Every constant, path, label map, and plotting default lives here so that no
magic number is repeated across modules. Import what you need:

    from irdpfn import config as C
    C.WINDOW, C.REGIME_NAMES, C.OUTPUTS_DIR, ...
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (resolved relative to the repository root, two levels up from here)
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parents[1]

DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"
FIGURES_DIR = ROOT_DIR / "figures"
OUTPUTS_DIR = ROOT_DIR / "outputs"

SYNTHETIC_CSV = DATA_DIR / "pension_fund_synthetic.csv"

for _d in (DATA_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Panel dimensions
# ---------------------------------------------------------------------------
N = 40            # pension funds = 5 providers x 8 age cohorts
M = 5             # global benchmark factors (3 equity + 2 yield)
N_PROVIDERS = 5
N_COHORTS = 8

PROVIDERS = [f"Provider {i}" for i in range(1, N_PROVIDERS + 1)]
COHORTS = [f"AG{i}" for i in range(1, N_COHORTS + 1)]

# AG code -> human-readable age band shown in the AgeGroup column
COHORT_BANDS = {
    "AG1": "54/60",
    "AG2": "61/67",
    "AG3": "68/74",
    "AG4": "75/81",
    "AG5": "82/88",
    "AG6": "89/95",
    "AG7": "96/02",
    "AG8": "TIPF",
}

# Lifecycle segmentation used by the non-circularity check
CONSERVATIVE_COHORTS = {"AG1", "AG8"}   # youngest band + target / index pension fund
GROWTH_COHORTS = {"AG2", "AG3", "AG4", "AG5", "AG6", "AG7"}

# ---------------------------------------------------------------------------
# Rolling-window / absorption-ratio settings
# ---------------------------------------------------------------------------
WINDOW = 60                     # baseline rolling window (trading days)
WINDOWS = [40, 60, 90, 120]     # window-length robustness sweep
RIDGE_EPS = 1e-6                # only applied when N > window (augmented matrix)

# ---------------------------------------------------------------------------
# HMM / regime settings
# ---------------------------------------------------------------------------
SEED = 2026
K_BASELINE = 3
K_RANGE_BIC = range(2, 9)
K_RANGE_NUTS = [3, 4, 5, 6, 7]
KAPPAS_SENS = [10, 15, 24, 66, 80]

MCMC_BASELINE = {"num_warmup": 1000, "num_samples": 1000, "num_chains": 4}
MCMC_FAST = {"num_warmup": 500, "num_samples": 500, "num_chains": 2}

REGIME_NAMES = [
    "Low-Concentration",       # state 0  (lowest emission mean)
    "Moderate-Concentration",  # state 1
    "High-Concentration",      # state 2  (crisis)
]

# ---------------------------------------------------------------------------
# Global-factor tickers (used only when fetching REAL data in data_io)
# ---------------------------------------------------------------------------
GLOBAL_TICKERS = {"MSCI_World": "URTH", "MSCI_Europe": "IEUR", "SP500": "^GSPC"}
ECB_YIELD_KEYS = {
    "dY_10Y": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
    "dY_2Y": "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
}
EQUITY_COLS = ["MSCI_World", "MSCI_Europe", "SP500"]
YIELD_COLS = ["dY_10Y", "dY_2Y"]
GLOBAL_COLS = EQUITY_COLS + YIELD_COLS

# ---------------------------------------------------------------------------
# Crisis / regime event windows  (shading + crisis-vs-calm split)
# ---------------------------------------------------------------------------
EVENTS = {
    "COVID-19 crash":                 ("2020-02-19", "2020-03-23", "#d73027", 0.18),
    "Inflation / supply-chain shock": ("2022-01-01", "2022-06-16", "#fdae61", 0.08),
    "Russia-Ukraine war":             ("2022-02-24", "2022-03-08", "#4575b4", 0.15),
    "Banking stress":                 ("2023-03-08", "2023-03-31", "#4daf4a", 0.20),
}

EVENT_LABELS = [
    ("2020-03-01", "COVID-19", "#d73027"),
    ("2022-03-03", "Ukraine War", "#4575b4"),
    ("2022-05-16", "Inflation shock\n(discrete crisis)", "#fdae61"),
    ("2023-03-20", "Banking\nStress", "#4daf4a"),
]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
PROVIDER_COLORS = {
    "Provider 1": "#1f77b4",
    "Provider 2": "#ff7f0e",
    "Provider 3": "#2ca02c",
    "Provider 4": "#d62728",
    "Provider 5": "#9467bd",
}

REGIME_COLORS = {
    "Low-Concentration": "#2166ac",
    "Moderate-Concentration": "#fdae61",
    "High-Concentration": "#d73027",
}

EQUITY_SERIES = {
    "MSCI_World":  ("#1f77b4", "-",  1.8, "MSCI World"),
    "MSCI_Europe": ("#ff7f0e", "--", 1.8, "MSCI Europe"),
    "SP500":       ("#2ca02c", "-.", 1.8, "S&P 500"),
}
RATE_SERIES = {
    "dY_10Y": ("#d62728", "-",  1.6, r"$\Delta Y_{10Y}$"),
    "dY_2Y":  ("#9467bd", "--", 1.6, r"$\Delta Y_{2Y}$"),
}

# ---------------------------------------------------------------------------
# Matplotlib style applied once via figures.set_style()
# ---------------------------------------------------------------------------
PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linestyle": "--",
    "figure.dpi": 150,
}
