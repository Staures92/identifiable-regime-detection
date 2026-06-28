"""
Data input/output and matrix construction.

Loads the (synthetic or real) long panel, pivots it into the fund return
matrix R^(f) and the fund-specific benchmark matrix R^(b,f), obtains the
global-factor matrix G^(b) (either fetched live or generated offline), and
aligns everything onto a common trading calendar, returning the augmented
matrix R_aug = [R^(f) | R^(b,f) | G^(b)].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import synthetic_data as synth


@dataclass
class Panel:
    """Container for every aligned matrix the pipeline consumes."""
    R_f: pd.DataFrame      # [T x N]      fund returns
    R_bf: pd.DataFrame     # [T x N]      fund-specific benchmark returns
    G_b: pd.DataFrame      # [T x M]      global factors (equity + yields)
    R_aug: pd.DataFrame    # [T x (2N+M)] augmented matrix
    raw: pd.DataFrame      # original long panel


# ---------------------------------------------------------------------------
# Long panel -> wide matrices
# ---------------------------------------------------------------------------
def load_long_panel(path=C.SYNTHETIC_CSV) -> pd.DataFrame:
    """Read the long panel; (re)generate the synthetic file if it is absent."""
    if not path.exists():
        synth.write_synthetic_csv(path)
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Provider", "Date"]).reset_index(drop=True)


def build_return_matrices(df: pd.DataFrame):
    """Pivot the long panel into R^(f) and R^(b,f) (both [T x N])."""
    R_f = df.pivot_table(index="Date", columns="Provider",
                         values="log_return_price", aggfunc="first").sort_index()
    R_bf = df.pivot_table(index="Date", columns="Provider",
                          values="log_return_index", aggfunc="first").sort_index()
    return R_f, R_bf


# ---------------------------------------------------------------------------
# Global factors
# ---------------------------------------------------------------------------
def fetch_global_factors(start: str, end: str) -> pd.DataFrame:
    """
    Live global factors: yfinance equity benchmarks + ECB AAA yield shocks.
    Imported lazily so the offline path never needs network libraries.
    """
    import requests
    from io import StringIO
    import yfinance as yf

    prices = yf.download(list(C.GLOBAL_TICKERS.values()), start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
    prices.columns = list(C.GLOBAL_TICKERS.keys())
    R_be = np.log(prices / prices.shift(1)).dropna()
    R_be.index = pd.to_datetime(R_be.index)

    def _ecb(series_key):
        url = (f"https://data-api.ecb.europa.eu/service/data/YC/{series_key}"
               f"?startPeriod={start}&endPeriod={end}&format=csvdata")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        raw = pd.read_csv(StringIO(r.text))
        return (raw[["TIME_PERIOD", "OBS_VALUE"]]
                .assign(TIME_PERIOD=lambda x: pd.to_datetime(x["TIME_PERIOD"]))
                .set_index("TIME_PERIOD")["OBS_VALUE"].sort_index())

    y10 = _ecb(C.ECB_YIELD_KEYS["dY_10Y"])
    y2 = _ecb(C.ECB_YIELD_KEYS["dY_2Y"])
    rates = pd.concat([(y10.diff() * 100).rename("dY_10Y"),
                       (y2.diff() * 100).rename("dY_2Y")], axis=1).dropna()
    return pd.concat([R_be, rates], axis=1).dropna()


def get_global_factors(start, end, source: str = "synthetic") -> pd.DataFrame:
    """Dispatch to synthetic (default, offline) or live factor construction."""
    if source == "synthetic":
        return synth.generate_global_factors(start=start, end=end)
    if source == "live":
        return fetch_global_factors(start, end)
    raise ValueError("source must be 'synthetic' or 'live'")


# ---------------------------------------------------------------------------
# Alignment / augmentation
# ---------------------------------------------------------------------------
def align(R_f, R_bf, G_b) -> Panel:
    """Restrict all matrices to common dates and build R_aug."""
    common = R_f.index.intersection(R_bf.index).intersection(G_b.index)
    R_f = R_f.loc[common].ffill().dropna()
    R_bf = R_bf.loc[common].ffill().dropna()
    G_b = G_b.loc[common].ffill().dropna()
    common = R_f.index.intersection(R_bf.index).intersection(G_b.index)
    R_f, R_bf, G_b = R_f.loc[common], R_bf.loc[common], G_b.loc[common]
    R_aug = pd.concat([R_f, R_bf, G_b], axis=1).dropna()
    return Panel(R_f=R_f, R_bf=R_bf, G_b=G_b, R_aug=R_aug, raw=None)


def load_panel(path=C.SYNTHETIC_CSV, factor_source: str = "synthetic") -> Panel:
    """One-call loader: long panel -> aligned Panel ready for the pipeline."""
    df = load_long_panel(path)
    R_f, R_bf = build_return_matrices(df)
    start = df["Date"].min().strftime("%Y-%m-%d")
    end = df["Date"].max().strftime("%Y-%m-%d")
    G_b = get_global_factors(start, end, source=factor_source)
    panel = align(R_f, R_bf, G_b)
    panel.raw = df
    return panel


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------
def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-age-band and full-sample moments of the price log-returns."""
    rows = []
    for band, data in df.groupby("AgeGroup"):
        s = data["log_return_price"].dropna()
        rows.append({"Series": band, "Mean": s.mean(), "Std": s.std(),
                     "Min": s.min(), "Max": s.max(),
                     "Skewness": s.skew(), "Kurtosis": s.kurt()})
    s = df["log_return_price"].dropna()
    rows.append({"Series": "Full sample", "Mean": s.mean(), "Std": s.std(),
                 "Min": s.min(), "Max": s.max(),
                 "Skewness": s.skew(), "Kurtosis": s.kurt()})
    return pd.DataFrame(rows).set_index("Series").round(4)


def factor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile + moment summary for any wide factor frame."""
    return pd.DataFrame({
        "Mean": df.mean(), "Std": df.std(), "Min": df.min(),
        "25%": df.quantile(0.25), "Median": df.median(),
        "75%": df.quantile(0.75), "Max": df.max(),
        "Skew": df.skew(), "Kurtosis": df.kurtosis(),
    })
