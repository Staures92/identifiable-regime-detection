"""
Step 00 — Generate the synthetic pension-fund panel.

The empirical study uses proprietary daily NAV data for the Lithuanian
second-pillar pension funds, which cannot be redistributed. This script writes
a *public stand-in* (``data/pension_fund_synthetic.csv``) that reproduces the
statistical structure the pipeline relies on: a cohort volatility ladder,
fat tails, negative skew, crisis-clustered concentration spikes, and an
augmented absorption ratio that sits *below* the baseline. It is a structural
substitute, not the original data, so absolute numbers differ from the paper.

Run:  python scripts/00_generate_synthetic_data.py
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

from irdpfn import synthetic_data, data_io, config as C


def main():
    print("=" * 70)
    print("STEP 00 — Synthetic pension-fund panel")
    print("=" * 70)

    df = synthetic_data.write_synthetic_csv(C.SYNTHETIC_CSV, seed=C.SEED)
    print(f"\nWrote {len(df):,} long rows -> {C.SYNTHETIC_CSV}")
    print(f"Funds: {df['Provider'].nunique()}   "
          f"Dates: {df['Date'].min().date()} -> {df['Date'].max().date()}")

    print("\nPer-cohort summary (log_return_price):")
    print(data_io.cohort_summary(df).to_string())

    print("\nStep 00 complete.")


if __name__ == "__main__":
    main()
