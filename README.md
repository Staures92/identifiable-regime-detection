# Identifiable Regime Detection in Pension Fund Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20408012.svg)](https://doi.org/10.5281/zenodo.20408012)

Reference implementation of the absorption-ratio → DTW-clustering → sticky-HMM
regime-detection pipeline from the paper *"Identifiable Regime Detection in
Pension Fund Networks via Sticky Hidden Markov Models"*.

The empirical study uses proprietary daily NAV data for the Lithuanian
second-pillar pension funds, which cannot be redistributed. This repository
ships a **public synthetic panel** that reproduces the statistical structure the
method relies on, so the full pipeline can be run and inspected end to end on
data that is safe to share. The synthetic data is a structural stand-in, **not**
a reproduction of the paper's numbers — see
[`docs/methodology_notes.md`](docs/methodology_notes.md) §5 for the caveats.

---

## Install

```bash
git clone https://github.com/Staures92/identifiable-regime-detection
cd identifiable-regime-detection
pip install -e .            # core pipeline
pip install -e ".[bayes]"   # + NUTS identifiability sweep (optional, slow)
pip install -e ".[test]"    # + pytest
```

Python ≥ 3.10. The scripts also run without installation (each adds `src/` to the
path), so `python scripts/01_absorption_ratio.py` works straight after cloning.

## Run

```bash
python scripts/run_all.py            # fast: synthetic data, no NUTS  (~5 min)
python scripts/run_all.py --full     # paper-grade settings + NUTS    (slow)
```

or step by step:

```bash
python scripts/00_generate_synthetic_data.py   # writes data/pension_fund_synthetic.csv
python scripts/01_absorption_ratio.py          # AR_t, estimators, lead-lag
python scripts/02_clustering.py                # DTW clusters, non-circularity
python scripts/03_regime_detection.py          # EM-HMM, threshold τ, robustness
python scripts/04_diagnostics.py               # correlations, regime OLS, MAAR
```

Figures land in `figures/`, tables in `outputs/`. The annotated walk-through is
[`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb).

Heavy options are flags, so the default run is fast and offline:

```bash
python scripts/02_clustering.py --n-init 20 --max-iter 100   # paper k-means budget
python scripts/03_regime_detection.py --bayes --n-seeds 200  # NUTS + more EM restarts
```

## Test

```bash
pytest -q
```

---

## Repository layout

```
identifiable-regime-detection/
├── data/                      # synthetic panel (generated)
├── docs/methodology_notes.md  # decisions + synthetic-data caveats
├── figures/                   # generated PDFs (synthetic data)
├── paper_figures/             # real-data reference figures from the manuscript
├── outputs/                   # generated CSVs
├── notebooks/pipeline.ipynb   # annotated end-to-end walk-through
├── scripts/                   # 00–04 + run_all.py (thin orchestrators)
├── src/irdpfn/                # the package
│   ├── config.py              # paths, constants, plotting style, event windows
│   ├── synthetic_data.py      # public stand-in generator
│   ├── data_io.py             # panel assembly, R_f / R_bf / G_b / R_aug
│   ├── absorption.py          # absorption ratio, estimators, lead-lag
│   ├── clustering.py          # DTW clustering + non-circularity
│   ├── regime.py              # EM-HMM, thresholds, robustness, sticky NUTS
│   ├── diagnostics.py         # correlations, regime OLS, MAAR amplification
│   └── figures.py             # all figures
└── tests/test_pipeline.py     # smoke + property tests
```

`paper_figures/` holds the real-data reference figures from the manuscript.
There are two views of the absorption ratio with its benchmarks, both
regenerated from the synthetic panel under `figures/`:
`fig02_ar_benchmarks_states_*` shades AR_t by the three detected HMM states
(with τ = 0.845, the 1/N floor, and the 37.5% crisis-day count), while
`fig02_ar_benchmarks_events_*` shades it by the four labelled crisis episodes
(COVID-19, Ukraine war, inflation shock, banking stress) and shows the raw yield
shocks. Only the underlying values differ between `paper_figures/` (real) and
`figures/` (synthetic).

## Method at a glance

| Stage | What it does | Key choices |
| --- | --- | --- |
| Absorption ratio | PC1 variance share over a 60-day window | covariance **and** scale-free correlation form; augmented `A = [R_f \| R_bf \| G_b]`, N = 85 |
| Clustering | DTW dissimilarity + average linkage | silhouette in DTW geometry → `K*`; non-circularity vs label partitions |
| Regime detection | Gaussian EM-HMM on `AR_t` | BIC over K; `K = 3` baseline; crisis threshold τ from equal-density Gaussian crossing |
| Identifiability | sticky HMM via NUTS | κ calibrated from EM persistence; `K = 3` posterior-identifiable |
| Diagnostics | interpret + stress-test | regime-conditional OLS (HAC), augmented agreement + permutation, per-cluster MAAR |

Full detail, including the overlap-window defence and the synthetic-data
limitations, is in [`docs/methodology_notes.md`](docs/methodology_notes.md).

## Citing

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata, or use GitHub's
"Cite this repository" button. If you use this code, please cite both the
manuscript and the software.

**Manuscript:**
> Megang Nkamga, J. S., & Kabašinskas, A. (2026). Identifiable Regime Detection
> in Pension Fund Networks via Sticky Hidden Markov Models. Working paper,
> Kaunas University of Technology.

**Software:**
> Megang Nkamga, J. S., & Kabašinskas, A. (2026). Identifiable Regime Detection
> in Pension Fund Networks via Sticky Hidden Markov Models [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.20408012

The Zenodo link is the concept DOI; it always resolves to the latest archived
version.

## License

MIT — see [`LICENSE`](LICENSE). The synthetic data carries the same license; the
underlying real pension-fund NAV data is not included and is not redistributable.
