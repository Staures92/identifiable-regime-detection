"""
Step 00 — Generate synthetic pension fund panel.

Creates `data/pension_fund_synthetic.csv` with 5 providers x 8 age
cohorts x ~7 years of daily returns, embedding three regimes so that
the downstream pipeline produces meaningful figures.

Run from the project root:
    python scripts/00_generate_synthetic_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irdpfn.synthetic_data import generate_panel


def main():
    df, path = generate_panel()
    print(f"Wrote {len(df):,} rows to {path}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Funds: {df['Provider'].nunique()} providers "
          f"x {df['AgeGroup'].nunique()} cohorts "
          f"= {df['Provider'].nunique() * df['AgeGroup'].nunique()}")


if __name__ == "__main__":
    main()
