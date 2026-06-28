"""
DTW clustering of fund return dynamics.

Standardise each fund's return series, build the DTW dissimilarity matrix,
cluster with average linkage, and select K* by the DTW-geometry silhouette.
Robustness spans algorithm (k-means) and metric (correlation, raw Euclidean);
a non-circularity check shows the recovered clusters track return co-movement
rather than merely echoing the provider x cohort labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from dtaidistance import dtw
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

from . import config as C


@dataclass
class ClusterResult:
    fund_names: list
    dtw_matrix: np.ndarray
    linkage: np.ndarray
    silhouette: pd.DataFrame
    best_k: int
    labels_hier: np.ndarray
    labels_dtw: np.ndarray = None
    labels_corr: np.ndarray = None
    labels_euc: np.ndarray = None
    ari: dict = field(default_factory=dict)
    noncircularity: pd.DataFrame = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cluster_te(R_f, R_bf, funds):
    """Mean absolute active return (tracking error) over a set of funds."""
    return (R_f[funds] - R_bf[funds]).abs().mean().mean()


def cluster_composition(labels, fund_names, R_f, R_bf) -> pd.DataFrame:
    """Per-cluster fund membership and mean tracking error."""
    rows = []
    for cid in sorted(np.unique(labels)):
        funds = [fund_names[i] for i in np.where(labels == cid)[0]]
        rows.append({"Cluster": int(cid), "n_funds": len(funds),
                     "mean_TE": round(_cluster_te(R_f, R_bf, funds), 6),
                     "funds": ", ".join(funds)})
    return pd.DataFrame(rows)


def _run_kmeans(data, metric, k, seed=C.SEED, n_init=20, max_iter=100):
    km = TimeSeriesKMeans(n_clusters=k, metric=metric, n_init=n_init,
                          max_iter=max_iter, random_state=seed, n_jobs=-1)
    return km.fit_predict(data)


def _encode(values):
    keys = {v: i for i, v in enumerate(sorted(set(values)))}
    return np.array([keys[v] for v in values])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_clustering(R_f, R_bf, k_min=2, k_max=15,
                   robustness=True, noncircularity=True,
                   kmeans_n_init=20, kmeans_max_iter=100) -> ClusterResult:
    fund_names = R_f.columns.tolist()
    n = len(fund_names)

    R_f_scaled = StandardScaler().fit_transform(R_f)          # T x N, z-scored
    series_std = R_f_scaled.T                                 # N x T

    dtw_matrix = dtw.distance_matrix_fast(series_std)
    np.fill_diagonal(dtw_matrix, 0.0)
    Z = linkage(squareform(dtw_matrix), method="average")

    sil_rows = []
    for k in range(k_min, k_max + 1):
        labels_k = fcluster(Z, k, criterion="maxclust")
        sil = silhouette_score(dtw_matrix, labels_k, metric="precomputed")
        sil_rows.append({"K": k, "DTW_Silhouette": sil})
    silhouette = pd.DataFrame(sil_rows)
    best_k = int(silhouette.loc[silhouette["DTW_Silhouette"].idxmax(), "K"])

    labels_hier = fcluster(Z, best_k, criterion="maxclust")

    res = ClusterResult(fund_names=fund_names, dtw_matrix=dtw_matrix, linkage=Z,
                        silhouette=silhouette, best_k=best_k,
                        labels_hier=labels_hier)

    if robustness:
        ts_std = to_time_series_dataset([series_std[i] for i in range(n)])
        ts_raw = to_time_series_dataset([R_f.values.T[i] for i in range(n)])
        res.labels_dtw = _run_kmeans(ts_std, "dtw", best_k,
                                     n_init=kmeans_n_init, max_iter=kmeans_max_iter)
        # correlation k-means == Euclidean on standardised series
        res.labels_corr = _run_kmeans(ts_std, "euclidean", best_k,
                                      n_init=kmeans_n_init, max_iter=kmeans_max_iter)
        res.labels_euc = _run_kmeans(ts_raw, "euclidean", best_k,
                                     n_init=kmeans_n_init, max_iter=kmeans_max_iter)
        res.ari = {
            "DTW(avg-link) vs DTW k-means": adjusted_rand_score(labels_hier, res.labels_dtw),
            "DTW k-means vs Correlation":   adjusted_rand_score(res.labels_dtw, res.labels_corr),
            "DTW k-means vs Euclidean(raw)": adjusted_rand_score(res.labels_dtw, res.labels_euc),
            "Correlation vs Euclidean(raw)": adjusted_rand_score(res.labels_corr, res.labels_euc),
        }

    if noncircularity:
        res.noncircularity = noncircularity_table(res)

    return res


def noncircularity_table(res: ClusterResult) -> pd.DataFrame:
    """ARI of return-based clusters against label-only reference partitions."""
    providers = [f.split("_")[0] for f in res.fund_names]
    cohorts = [f.split("_")[1] for f in res.fund_names]
    segment = ["Conservative" if c in C.CONSERVATIVE_COHORTS else "Growth"
               for c in cohorts]

    refs = {
        "Provider only (5)": _encode(providers),
        "Lifecycle segment (2)": _encode(segment),
        "Cohort only (8)": _encode(cohorts),
        "Provider x segment (10)": _encode(list(zip(providers, segment))),
    }
    rows = []
    for name, ref in refs.items():
        row = {"Reference partition": name,
               "Hierarchical": round(adjusted_rand_score(ref, res.labels_hier), 4)}
        if res.labels_dtw is not None:
            row["DTW k-means"] = round(adjusted_rand_score(ref, res.labels_dtw), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_frame(res: ClusterResult, which: str = "hier") -> pd.DataFrame:
    """Tidy Fund/Provider/AgeCohort/Cluster frame for figures and MAAR tables."""
    labels = {"hier": res.labels_hier, "dtw": res.labels_dtw}[which]
    return pd.DataFrame({
        "Fund": res.fund_names,
        "Provider": [f.split("_")[0] for f in res.fund_names],
        "AgeCohort": [f.split("_")[1] for f in res.fund_names],
        "Cluster": labels,
    })
