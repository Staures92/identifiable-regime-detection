"""
Run the full pipeline end to end (steps 00 -> 04).

By default the heavy options are off: the DTW k-means robustness uses a small
restart budget and the Bayesian NUTS sweep is skipped, so a complete run
finishes in minutes on the public synthetic panel. Pass ``--full`` for
paper-grade settings (more k-means restarts, more EM seeds, and the NUTS
identifiability sweep).

Run:  python scripts/run_all.py [--full]
"""
import argparse
import importlib
import sys
import warnings

warnings.filterwarnings("ignore")

# scripts use numeric prefixes, so import them by file path via runpy-style names
import runpy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS.parent / "src"))


def _run(stem, argv):
    sys.argv = [stem] + argv
    runpy.run_path(str(SCRIPTS / f"{stem}.py"), run_name="__main__")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="paper-grade settings (slow): full k-means + EM + NUTS")
    args = p.parse_args()

    _run("00_generate_synthetic_data", [])

    if args.full:
        _run("01_absorption_ratio", [])
        _run("02_clustering", ["--n-init", "20", "--max-iter", "100"])
        _run("03_regime_detection", ["--n-seeds", "200", "--bayes"])
        _run("04_diagnostics", [])
    else:
        _run("01_absorption_ratio", [])
        _run("02_clustering", [])
        _run("03_regime_detection", [])
        _run("04_diagnostics", [])

    print("\n" + "=" * 70)
    print("Pipeline complete. See figures/ and outputs/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
