"""
Step 02 — DTW hierarchical clustering with robustness checks.

Outputs:
    step02_db_scores_ward.csv
    step02_cluster_composition_K{star}.csv
    step02_robustness_kmeans_euclidean.csv
    step02_robustness_kmeans_dtw.csv
    step02_cross_method_ari.csv
    step02_nested_hierarchy.csv

Figures:
    fig05_dendrogram.pdf
    fig06_cluster_heatmap.pdf
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irdpfn.clustering import (
    cluster_composition,
    compute_dtw_linkage,
    cross_method_ari,
    db_score_sweep_ward,
    nested_hierarchy,
    robustness_kmeans_dtw,
    robustness_kmeans_euclidean,
)
from irdpfn.config import K_CLUSTER_STAR, OUTPUTS_DIR
from irdpfn.data_io import load_and_align
from irdpfn.figures import fig05_dendrogram, fig06_cluster_heatmap


def main():
    # 1. Load and align
    df, R_f, R_bf, R_bg, R_aug = load_and_align()
    print(f"Loaded panel: {df.shape}")

    # 2. DTW + Ward
    print("\nComputing DTW distance matrix and Ward linkage...")
    Z, R_f_scaled, fund_names = compute_dtw_linkage(R_f)

    # 3. DB score sweep
    db = db_score_sweep_ward(Z, R_f_scaled)
    db_df = pd.DataFrame({"K": list(db.keys()),
                          "DB_ward_dtw": list(db.values())})
    db_df.to_csv(OUTPUTS_DIR / "step02_db_scores_ward.csv", index=False)
    print("\nDavies-Bouldin (Ward + DTW):")
    print(db_df.to_string(index=False))

    K_star = min(db, key=db.get)
    print(f"\nOptimal K_c* = {K_star}  (configured: {K_CLUSTER_STAR})")

    # 4. Cluster composition
    cluster_df, comp_summary = cluster_composition(
        Z, fund_names, R_f, R_bf, K_CLUSTER_STAR,
    )
    cluster_df.to_csv(
        OUTPUTS_DIR / f"step02_cluster_assignments_K{K_CLUSTER_STAR}.csv",
        index=False,
    )
    comp_summary[["Cluster", "N_funds", "Mean_TE"]].to_csv(
        OUTPUTS_DIR / f"step02_cluster_composition_K{K_CLUSTER_STAR}.csv",
        index=False,
    )
    print(f"\nCluster composition (K={K_CLUSTER_STAR}):")
    for _, row in comp_summary.iterrows():
        print(f"  C{row['Cluster']} (n={row['N_funds']}, "
              f"TE={row['Mean_TE']:.6f}): {row['Funds']}")

    # 5. Robustness — k-means Euclidean
    print("\nRobustness 1: K-means (Euclidean)")
    rob_eucl = robustness_kmeans_euclidean(R_f_scaled, Z)
    rob_eucl.to_csv(OUTPUTS_DIR / "step02_robustness_kmeans_euclidean.csv",
                    index=False)
    print(rob_eucl.to_string(index=False))

    # 6. Robustness — k-means DTW
    # NOTE: TimeSeriesKMeans with DTW barycenters is O(T^2) per iteration
    # per pairwise alignment; we subsample to 200 dates to keep it tractable.
    print("\nRobustness 2: K-means (DTW barycenters) — subsampled to 200 dates")
    rob_dtw, ts_data = robustness_kmeans_dtw(
        R_f_scaled, Z, K_range=range(2, 9),
        max_iter=10, n_init=2, subsample_dates=200,
    )
    rob_dtw.to_csv(OUTPUTS_DIR / "step02_robustness_kmeans_dtw.csv",
                   index=False)
    print(rob_dtw.to_string(index=False))

    # 7. ARI across methods
    ari = cross_method_ari(Z, R_f_scaled, ts_data, k=K_CLUSTER_STAR)
    ari_df = pd.DataFrame([
        {"comparison": "ward_vs_kmeans_eucl",
         "ARI": ari["ward_vs_kmeans_eucl"]},
        {"comparison": "ward_vs_kmeans_dtw",
         "ARI": ari["ward_vs_kmeans_dtw"]},
        {"comparison": "kmeans_eucl_vs_dtw",
         "ARI": ari["kmeans_eucl_vs_dtw"]},
    ])
    ari_df.to_csv(OUTPUTS_DIR / "step02_cross_method_ari.csv", index=False)
    print(f"\nAdjusted Rand Index across methods (K={K_CLUSTER_STAR}):")
    print(ari_df.to_string(index=False))

    # 8. Nested K=8 -> K=15
    nested, n_singletons = nested_hierarchy(
        Z, fund_names, R_f, R_bf, k_outer=8, k_inner=15,
    )
    nested.to_csv(OUTPUTS_DIR / "step02_nested_hierarchy.csv", index=False)
    print(f"\nK=15 singletons: {n_singletons}/15")

    # 9. Figures
    fig05_dendrogram(Z, fund_names, k_star=K_CLUSTER_STAR)
    fig06_cluster_heatmap(cluster_df, k_star=K_CLUSTER_STAR)

    print("\nStep 02 complete.")


if __name__ == "__main__":
    main()
