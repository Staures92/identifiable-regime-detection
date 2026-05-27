# Methodology Notes

Supplementary notes on implementation choices that aren't obvious from the code.

## Absorption ratio: regularisation policy

The baseline absorption ratio is computed on `R_f` (40 funds, 60-day window). Since `N = 40 < omega = 60`, the sample covariance matrix is full-rank and no regularisation is needed. Adding `epsilon * I` to the baseline materially distorts `AR_t` (differences of 0.01 to 0.04 across windows — see `outputs/step01_regularisation_verification.csv`).

The augmented matrix `R_aug = [R_f | R_bf | R_bg]` has 83 columns. With `omega = 60`, the covariance matrix is rank-deficient and `epsilon * I` is required. The code in `irdpfn.absorption.compute_absorption_ratio` switches on this automatically by checking `N > window`.

## Clustering: why Ward + DTW over k-means

K-means with DTW barycenters is theoretically problematic because DTW is a semi-metric (it satisfies positivity and symmetry but fails the triangle inequality). Centroids in DTW space are not well-defined, and the iteration is not guaranteed to converge to a stable solution. Ward linkage requires only a pairwise dissimilarity matrix and extends naturally to non-Euclidean metrics (Randriamihamison et al., 2020).

The robustness checks in `irdpfn.clustering` confirm this empirically: Ward + DTW achieves equal or better Davies-Bouldin scores than both k-means (Euclidean) and TimeSeriesKMeans (DTW barycenters) across the full K range. The Adjusted Rand Index between methods at K = 8 is reported in `outputs/step02_cross_method_ari.csv`.

## HMM: kappa calibration

The sticky HMM prior uses a Dirichlet distribution with concentration `alpha * 1 + kappa * e_k` for row `k` of the transition matrix. The expected self-transition probability is

```
E[pi_kk] = (alpha + kappa) / (K * alpha + kappa)
```

With `alpha = 1`, inverting this gives `kappa = (K * p - 1) / (1 - p)` where `p` is the target persistence. The code calibrates `p` from the diagonals of the EM-fitted transition matrix at K = 3 and K = 5, giving two data-driven kappa values that are validated against each other and against a sensitivity grid (`kappa in {10, 20, 30, 50}`).

## HMM: identifiability under label switching

Bayesian HMMs are subject to label switching across chains and draws: the same regime may be labelled differently in different chains. Raw R-hat values are therefore unreliable. The code in `irdpfn.regime.relabel_samples_by_chain` sorts emission means within each `(chain, draw)` and recomputes R-hat on the relabelled samples. A regime is considered identified if its corrected R-hat is below 1.1.

## Diagnostics: HAC standard errors

The regime-conditional regression uses Newey-West HAC standard errors to account for the autocorrelation in daily AR_t. The lag length is set by the rule of thumb `L = floor(4 * (T/100)^(2/9))` (Newey and West, 1987).

## Synthetic data generator

The generator in `irdpfn.synthetic_data` is calibrated to reproduce the qualitative features that matter for the pipeline:

- A sticky 3-state regime chain drives factor volatility and idiosyncratic volatility.
- Factor loadings are monotone in age cohort (AG1 conservative, AG8 aggressive) with provider-level noise.
- Fund-specific benchmarks track the funds with tracking error that scales with the regime.

It is NOT calibrated to reproduce the exact empirical values from the real data. Reviewers running the synthetic pipeline should expect figures that look qualitatively similar in shape (three identifiable regimes, cluster structure by cohort, etc.) but with different numerical values.
