"""
Data I/O.

Loads the pension fund panel, pivots it into return matrices, downloads
the global benchmarks from Yahoo Finance, and aligns everything on a
common set of trading dates.

Return matrices
---------------
R_f   : [T x N]      log returns of pension fund NAVs
R_bf  : [T x N]      log returns of each fund's own comparative index
R_bg  : [T x M]      log returns of global benchmarks (MSCI World, etc.)
R_aug : [T x (2N+M)] horizontal concatenation [R_f | R_bf | R_bg]
"""

import numpy as np
import pandas as pd
import yfinance as yf

from .config import DEFAULT_DATA_FILE, GLOBAL_TICKERS


# =========================================================
# 1. PENSION FUND PANEL
# =========================================================
def load_pension_panel(path=None):
    """
    Load the long-format pension fund panel.

    Expected columns: Date, Provider, AgeGroup, log_return_price,
                      log_return_index
    """
    path = path or DEFAULT_DATA_FILE
    df = pd.read_csv(path, parse_dates=["Date"])
    df = (
        df.sort_values(["Provider", "AgeGroup", "Date"])
          .reset_index(drop=True)
    )
    return df


def pivot_returns(df):
    """
    Pivot the long panel into wide return matrices.

    Each fund is identified by `Provider_AgeGroup` (e.g. "Provider 1_AG3"),
    so the resulting matrices have one column per fund.

    Returns
    -------
    R_f  : DataFrame [T x N]  fund returns
    R_bf : DataFrame [T x N]  fund-specific benchmark returns
    """
    df = df.copy()
    df["Fund"] = df["Provider"].astype(str) + "_" + df["AgeGroup"].astype(str)

    R_f = df.pivot_table(
        index="Date", columns="Fund",
        values="log_return_price", aggfunc="first",
    ).sort_index()

    R_bf = df.pivot_table(
        index="Date", columns="Fund",
        values="log_return_index", aggfunc="first",
    ).sort_index()

    return R_f, R_bf


# =========================================================
# 2. GLOBAL BENCHMARKS (Yahoo Finance)
# =========================================================
def download_global_benchmarks(start, end, tickers=None):
    """
    Download log-return series for the global benchmark tickers
    from Yahoo Finance.

    Parameters
    ----------
    start, end : str or Timestamp
    tickers    : dict mapping label -> Yahoo ticker (default GLOBAL_TICKERS)

    Returns
    -------
    R_bg : DataFrame [T x M]
    """
    tickers = tickers or GLOBAL_TICKERS
    raw = yf.download(
        list(tickers.values()),
        start=start, end=end,
        auto_adjust=True, progress=False,
    )["Close"]
    raw.columns = list(tickers.keys())

    R_bg = np.log(raw / raw.shift(1)).dropna()
    R_bg.index = pd.to_datetime(R_bg.index)
    return R_bg


# =========================================================
# 3. ALIGNMENT
# =========================================================
def align_all(R_f, R_bf, R_bg):
    """
    Restrict every return matrix to the common set of trading dates and
    build the augmented matrix R_aug = [R_f | R_bf | R_bg].
    """
    common = R_f.index.intersection(R_bf.index).intersection(R_bg.index)

    R_f   = R_f.loc[common].ffill().dropna()
    R_bf  = R_bf.loc[common].ffill().dropna()
    R_bg  = R_bg.loc[common].ffill().dropna()
    R_aug = pd.concat([R_f, R_bf, R_bg], axis=1)

    return R_f, R_bf, R_bg, R_aug


# =========================================================
# 4. CONVENIENCE: full pipeline
# =========================================================
def load_and_align(path=None):
    """One-shot loader: returns (df, R_f, R_bf, R_bg, R_aug)."""
    df = load_pension_panel(path)
    R_f, R_bf = pivot_returns(df)

    R_bg = download_global_benchmarks(
        start=df["Date"].min(),
        end=df["Date"].max(),
    )

    R_f, R_bf, R_bg, R_aug = align_all(R_f, R_bf, R_bg)
    return df, R_f, R_bf, R_bg, R_aug


# =========================================================
# 5. DESCRIPTIVE STATISTICS
# =========================================================
def descriptive_stats(returns):
    """
    Mean, std, min, max, skewness, excess kurtosis per column.
    Works on a DataFrame or a single Series.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()
    return pd.DataFrame({
        "Mean":           returns.mean(),
        "Std":            returns.std(ddof=1),
        "Min":            returns.min(),
        "Max":            returns.max(),
        "Skewness":       returns.skew(),
        "ExcessKurtosis": returns.kurtosis(),
    })
