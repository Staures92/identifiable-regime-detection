"""
Run the entire pipeline end-to-end.

Usage:
    python scripts/run_all.py

This calls steps 00-04 in order. Each step writes its own outputs and
figures, so it is safe to re-run a single step in isolation.
"""

import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

STEPS = [
    "00_generate_synthetic_data.py",
    "01_absorption_ratio.py",
    "02_clustering.py",
    "03_regime_detection.py",
    "04_diagnostics.py",
]


def main():
    for name in STEPS:
        path = SCRIPTS / name
        print(f"\n{'#' * 70}")
        print(f"# Running {name}")
        print(f"{'#' * 70}")
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT, check=False,
        )
        if result.returncode != 0:
            print(f"\n[!] {name} exited with code {result.returncode}")
            sys.exit(result.returncode)

    print("\n" + "=" * 70)
    print("Pipeline complete.")
    print("Figures: figures/")
    print("Outputs: outputs/")
    print("=" * 70)


if __name__ == "__main__":
    main()
