"""
Step 02 — DTW clustering of fund-level dynamics.

Standardises the fund returns, builds a DTW dissimilarity matrix, and forms an
average-linkage hierarchy. K* is selected by the silhouette computed in DTW
geometry. Robustness is assessed against DTW k-means and against
correlation / raw-Euclidean k-means (ARI). The non-circularity check shows the
return-based clusters recover the provider x lifecycle design without being a
label tautology.

DTW k-means dominates the runtime, so ``--n-init`` / ``--max-iter`` expose the
k-means restart budget. Defaults are modest so the script finishes quickly on
the public synthetic panel; raise them to reproduce paper-grade robustness.

Run:  python scripts/02_clustering.py [--n-init 20] [--max-iter 100]
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))

import argparse
import warnings

from irdpfn import data_io, clustering as cl, figures as fig, config as C

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-init", type=int, default=5,
                   help="k-means restarts (paper uses 20)")
    p.add_argument("--max-iter", type=int, default=30,
                   help="k-means max iterations (paper uses 100)")
    p.add_argument("--no-robustness", action="store_true",
                   help="skip the slow k-means robustness block")
    args = p.parse_args()

    print("=" * 70)
    print("STEP 02 — DTW clustering")
    print("=" * 70)

    panel = data_io.load_panel()
    res = cl.run_clustering(panel.R_f, panel.R_bf,
                            robustness=not args.no_robustness,
                            noncircularity=not args.no_robustness,
                            kmeans_n_init=args.n_init,
                            kmeans_max_iter=args.max_iter)

    print(f"\nBest K by DTW silhouette: K* = {res.best_k}")
    print(res.silhouette.to_string(index=False))

    if res.ari is not None:
        print("\nAlgorithm / distance-measure robustness (ARI):")
        for k, v in res.ari.items():
            print(f"  {k:<32} {v:.4f}")

    if res.noncircularity is not None:
        print("\nNon-circularity (ARI vs label-only partitions):")
        print(res.noncircularity.to_string(index=False))

    cluster_df = cl.cluster_frame(res, which="hier")
    print("\nCluster composition (hierarchical):")
    print(cl.cluster_composition(res.labels_hier, res.fund_names,
                                 panel.R_f, panel.R_bf).to_string(index=False))

    # --- persist + figures ---------------------------------------------------
    res.silhouette.to_csv(C.OUTPUTS_DIR / "step02_silhouette.csv", index=False)
    cluster_df.to_csv(C.OUTPUTS_DIR / "step02_clusters.csv", index=False)
    if res.noncircularity is not None:
        res.noncircularity.to_csv(C.OUTPUTS_DIR / "step02_noncircularity.csv",
                                  index=False)

    fig.fig_dendrogram(res)
    fig.fig_cluster_heatmap(res, which="dtw" if res.labels_dtw is not None else "hier")
    print(f"\nFigures + tables written to {C.FIGURES_DIR} and {C.OUTPUTS_DIR}")
    print("Step 02 complete.")


if __name__ == "__main__":
    main()
