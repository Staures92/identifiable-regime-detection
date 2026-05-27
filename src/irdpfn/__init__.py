"""
IRDPFN: Identifiable Regime Detection in Pension Fund Networks
================================================================

Replication package for:
    "Identifiable Regime Detection in Pension Fund Networks via
     Sticky Hidden Markov Models"

Modules
-------
config          : project-wide constants (windows, K-range, paths, colours)
data_io         : data loading, alignment, benchmark download
absorption      : absorption ratio + risk decomposition
clustering      : DTW + Ward hierarchical clustering (with robustness)
regime          : sticky HMM regime detection (EM + Bayesian validation)
diagnostics     : correlation tests, regime-conditional regression
figures         : all publication figures
"""

__version__ = "1.0.0"
__author__  = "Megang Nkamga Junile Staures"
