"""
irdpfn — Identifiable Regime Detection in Pension Fund networks.

Reference implementation accompanying the manuscript
"Identifiable Regime Detection in Pension Fund Networks via Sticky Hidden
Markov Models".

The package is organised as a thin set of composable modules:

    config        constants, paths, labels, plotting style
    synthetic_data generate a public stand-in for the proprietary NAV panel
    data_io       load the panel, build / align return + factor matrices
    absorption    rolling absorption ratio and its robustness variants
    clustering    DTW dissimilarity, hierarchy, silhouette K*, robustness
    regime        EM-HMM, crisis threshold, sticky Bayesian HMM, diagnostics
    diagnostics   correlation / regime-conditional / permutation checks
    figures       every manuscript figure behind shared helpers
"""
from . import config
from . import synthetic_data
from . import data_io
from . import absorption
from . import clustering
from . import regime
from . import diagnostics
from . import figures

__all__ = [
    "config", "synthetic_data", "data_io", "absorption",
    "clustering", "regime", "diagnostics", "figures",
]
__version__ = "1.1.0"
