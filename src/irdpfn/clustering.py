"""
Hierarchical clustering of pension funds.

Methodology
-----------
- Standardise R_f columns (z-score by fund).
- Compute pairwise DTW distance matrix.
- Apply Ward linkage; cut at K = K_c* = 8.

Robustness
----------
- K-means on standardised returns (Euclidean).
- TimeSeriesKMeans with DTW barycenters.
- Cross-method agreement via Adjusted Rand Index.
- Nested K=8 -> K=15 hierarchy.
"""

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from .config import K_CLUSTER_RANGE, K_CLUSTER_STAR


# =========================================================
# 1. DTW DISTANCE MATRIX + WARD LINKAGE
# =========================================================
def compute_dtw_linkage(R_f):
    """
    Returns
    -------
    Z          : Ward linkage matrix
    R_f_scaled : standardised returns (T x N)
    fund_names : column labels (length N)
    """
    scaler     = StandardScaler()
    R_f_scaled = scaler.fit_transform(R_f)
    fund_names = R_f.columns.tolist()

    dtw_matrix = dtw.distance_matrix_fast(R_f_scaled.T)
    np.fill_diagonal(dtw_matrix, 0)

    Z = linkage(squareform(dtw_matrix), method="ward")
    return Z, R_f_scaled, fund_names


# =========================================================
# 2. DB SCORE SWEEP (Ward + DTW)
# =========================================================
def db_score_sweep_ward(Z, R_f_scaled, K_range=K_CLUSTER_RANGE):
    """Davies-Bouldin score over K range for Ward+DTW labels."""
    db = {}
    for k in K_range:
        labels = fcluster(Z, k, criterion="maxclust")
        db[k]  = davies_bouldin_score(R_f_scaled.T, labels)
    return db


# =========================================================
# 3. CLUSTER COMPOSITION AND TRACKING ERROR
# =========================================================
def cluster_composition(Z, fund_names, R_f, R_bf, k):
    """Return per-cluster fund list and mean tracking error."""
    labels = fcluster(Z, k, criterion="maxclust")
    df = pd.DataFrame({"Fund": fund_names, "Cluster": labels})

    rows = []
    for c in sorted(df["Cluster"].unique()):
        funds = df[df["Cluster"] == c]["Fund"].tolist()
        te    = (R_f[funds] - R_bf[funds]).abs().mean().mean()
        rows.append({"Cluster": c, "N_funds": len(funds),
                     "Mean_TE": te, "Funds": funds})
    return df, pd.DataFrame(rows)


# =========================================================
# 4. ROBUSTNESS — K-means (Euclidean and DTW barycenters)
# =========================================================
def robustness_kmeans_euclidean(R_f_scaled, Z, K_range=range(2, 9)):
    """Compare Ward+DTW vs K-means (Euclidean) over K range."""
    rows = []
    for k in K_range:
        km     = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(R_f_scaled.T)
        db_km  = davies_bouldin_score(R_f_scaled.T, labels)
        db_w   = davies_bouldin_score(R_f_scaled.T,
                                      fcluster(Z, k, criterion="maxclust"))
        rows.append({"K": k, "DB_kmeans": db_km, "DB_ward": db_w,
                     "Ward_better": db_w < db_km})
    return pd.DataFrame(rows)


def robustness_kmeans_dtw(R_f_scaled, Z, K_range=range(2, 9),
                          max_iter=10, n_init=2, subsample_dates=None):
    """
    Compare Ward+DTW vs TimeSeriesKMeans (DTW barycenters).

    The DTW barycenter computation is O(T^2) per pairwise alignment per
    iteration, so for long series we optionally subsample dates uniformly.
    """
    if subsample_dates is not None and subsample_dates < R_f_scaled.shape[0]:
        idx = np.linspace(0, R_f_scaled.shape[0] - 1,
                          subsample_dates).astype(int)
        R_f_sub = R_f_scaled[idx]
    else:
        R_f_sub = R_f_scaled

    N = R_f_sub.shape[1]
    ts_data = to_time_series_dataset([R_f_sub.T[i, :] for i in range(N)])

    rows = []
    for k in K_range:
        km = TimeSeriesKMeans(
            n_clusters=k, metric="dtw", n_init=n_init,
            max_iter=max_iter, random_state=42, n_jobs=-1,
        )
        labels = km.fit_predict(ts_data)
        db_km  = davies_bouldin_score(R_f_sub.T, labels)
        db_w   = davies_bouldin_score(R_f_scaled.T,
                                      fcluster(Z, k, criterion="maxclust"))
        rows.append({"K": k, "DB_kmeans_dtw": db_km, "DB_ward": db_w,
                     "Ward_better": db_w < db_km})
    return pd.DataFrame(rows), ts_data


# =========================================================
# 5. ARI BETWEEN METHODS
# =========================================================
def cross_method_ari(Z, R_f_scaled, ts_data, k=K_CLUSTER_STAR,
                     max_iter=10, n_init=2):
    """Adjusted Rand Index between Ward, K-means Euclidean, K-means DTW."""
    ward = fcluster(Z, k, criterion="maxclust")
    km_e = KMeans(n_clusters=k, n_init=20, random_state=42)\
                .fit_predict(R_f_scaled.T)
    km_d = TimeSeriesKMeans(n_clusters=k, metric="dtw", n_init=n_init,
                            max_iter=max_iter, random_state=42, n_jobs=-1)\
                .fit_predict(ts_data)

    return {
        "ward_vs_kmeans_eucl": adjusted_rand_score(ward, km_e),
        "ward_vs_kmeans_dtw":  adjusted_rand_score(ward, km_d),
        "kmeans_eucl_vs_dtw":  adjusted_rand_score(km_e, km_d),
        "labels_ward":         ward,
        "labels_kmeans_eucl":  km_e,
        "labels_kmeans_dtw":   km_d,
    }


# =========================================================
# 6. NESTED HIERARCHY (K=8 → K=15)
# =========================================================
def nested_hierarchy(Z, fund_names, R_f, R_bf, k_outer=8, k_inner=15):
    """Examine how K_inner clusters nest inside K_outer clusters."""
    labels_outer = fcluster(Z, k_outer, criterion="maxclust")
    labels_inner = fcluster(Z, k_inner, criterion="maxclust")
    te_per_fund  = (R_f - R_bf).abs().mean()

    nested = pd.DataFrame({
        "Fund": fund_names,
        f"K{k_outer}":  labels_outer,
        f"K{k_inner}":  labels_inner,
        "TE":           [te_per_fund[f] for f in fund_names],
    })
    n_singletons = (pd.Series(labels_inner).value_counts() == 1).sum()
    return nested, n_singletons
