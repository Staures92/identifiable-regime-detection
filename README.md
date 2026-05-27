# Identifiable Regime Detection in Pension Fund Networks
[![DOI](https://zenodo.org/badge/1250935473.svg)](https://doi.org/10.5281/zenodo.20408012)

Replication code for *Identifiable Regime Detection in Pension Fund Networks via Sticky Hidden Markov Models*.


This repository computes a rolling-window absorption ratio on a panel of pension fund returns, identifies clusters of co-moving funds with DTW + Ward, and detects regime structure in the absorption ratio with a sticky Hidden Markov Model fit by EM and validated by Bayesian inference under label-switching-corrected identifiability diagnostics. The included synthetic dataset lets you run the full pipeline end-to-end without access to the real data.

## Repository layout

```
identifiable-regime-detection/
├── src/irdpfn/                  # Importable package
│   ├── config.py                  Project-wide constants and paths
│   ├── data_io.py                 Loading, pivoting, benchmark download
│   ├── absorption.py              AR_t, risk decomposition, alt. covariances
│   ├── clustering.py              DTW + Ward + robustness checks
│   ├── regime.py                  HMM (EM + Bayesian NUTS)
│   ├── diagnostics.py             Correlation tests, regime regression
│   ├── figures.py                 All publication figures
│   └── synthetic_data.py          Synthetic dataset generator
├── scripts/                    # Pipeline runners
│   ├── 00_generate_synthetic_data.py
│   ├── 01_absorption_ratio.py
│   ├── 02_clustering.py
│   ├── 03_regime_detection.py
│   ├── 04_diagnostics.py
│   └── run_all.py
├── data/                       # Input data (synthetic dataset written here)
├── figures/                    # Generated PDFs
├── outputs/                    # CSV tables and intermediate results
├── tests/                      # Smoke test
├── docs/                       # Supplementary notes
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Requires Python 3.14.3 or newer.

```bash
git clone <repo-url>
cd identifiable-regime-detection
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

JAX is installed in CPU mode by default. The Bayesian sweep in Step 03 runs in roughly 15-30 minutes on a modern CPU; on a GPU it is much faster. To use GPU JAX, follow the [official JAX installation guide](https://github.com/google/jax#installation).

## Running the pipeline

### Option 1 — Everything at once

```bash
python scripts/run_all.py
```

This runs steps 00 to 04 in order and writes all figures to `figures/` and tables to `outputs/`.

### Option 2 — One step at a time

```bash
python scripts/00_generate_synthetic_data.py   # creates data/pension_fund_synthetic.csv
python scripts/01_absorption_ratio.py          # AR_t, risk decomposition
python scripts/02_clustering.py                # DTW + Ward clustering
python scripts/03_regime_detection.py          # HMM regimes
python scripts/04_diagnostics.py               # cross-step diagnostics
```

To skip the slow Bayesian sweep in Step 03, open `scripts/03_regime_detection.py` and set `RUN_BAYESIAN = False`. The EM-based analysis, regime characterisation, and downstream diagnostics will still run.

## Using your own data

The pipeline expects a CSV with the columns `Date`, `Provider`, `AgeGroup`, `log_return_price`, `log_return_index`. To use real data instead of the synthetic dataset, place it at `data/pension_fund_real.csv` and update `DEFAULT_DATA_FILE` in `src/irdpfn/config.py`.

## Figures

All figures are numbered to match the paper.

| File | Description |
|---|---|
| `fig01_return_series.pdf` | Returns: R^(f), R^(b,f), R^(b,g) |
| `fig02_ar_benchmarks_events.pdf` | AR_t with event shading |
| `fig03_ar_benchmarks_regimes.pdf` | AR_t with HMM regime shading |
| `fig04_ar_baseline_vs_augmented.pdf` | Baseline vs augmented AR_t |
| `fig05_dendrogram.pdf` | Ward + DTW dendrogram |
| `fig06_cluster_heatmap.pdf` | Cluster heatmap (provider x cohort) |
| `fig07_emission_distributions.pdf` | HMM emission densities + thresholds |
| `fig08_cluster_regime_tracking_error.pdf` | Cluster x regime tracking error |
| `fig09_scatter_correlations.pdf` | AR_t vs benchmarks (scatter + OLS) |
| `fig10_regime_conditional_scatter.pdf` | Regime-conditional regression |
| `fig11_risk_decomposition.pdf` | Systematic vs specific risk |
| `fig12_covariance_comparison.pdf` | Sample vs Ledoit-Wolf vs MCD |

## Testing

A quick smoke test confirms the pipeline runs end-to-end on a small synthetic sample:

```bash
python -m pytest tests/
```

## Citation

If you use this code, please cite both the manuscript and the software:

**Manuscript:**
> Megang Nkamga, J. S., & Kabašinskas, A. (2026). Identifiable Regime Detection in Pension Fund Networks via Sticky Hidden Markov Models. Working paper, Kaunas University of Technology.

**Software:**
> Megang Nkamga, J. S., & Kabašinskas, A. (2026). Identifiable Regime Detection in Pension Fund Networks via Sticky Hidden Markov Models (v1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20408012


## License

MIT — see `LICENSE`.
