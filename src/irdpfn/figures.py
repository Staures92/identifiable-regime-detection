"""
All paper figures, de-duplicated behind shared style / shading helpers.

Every function takes already-computed objects and writes a PDF into
config.FIGURES_DIR, returning the path. Figure numbers in the docstrings map
to the manuscript.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from scipy.stats import norm as scipy_norm
from scipy.cluster.hierarchy import dendrogram, fcluster

from . import config as C


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def set_style():
    plt.rcParams.update(C.PLOT_STYLE)


def _save(fig, name):
    set_style.path = C.FIGURES_DIR / name
    fig.savefig(set_style.path, bbox_inches="tight")
    plt.close(fig)
    return set_style.path


def shade_events(ax, events=C.EVENTS):
    for _n, (s, e, c, a) in events.items():
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), color=c, alpha=a, lw=0, zorder=0)


def label_events(ax, labels=C.EVENT_LABELS, frac=0.01):
    ymin, ymax = ax.get_ylim()
    y = ymin + frac * (ymax - ymin)
    for x, txt, c in labels:
        ax.text(pd.Timestamp(x), y, txt, rotation=90, fontsize=7,
                color=c, ha="center", va="bottom", fontweight="bold")


def regime_shading(ax, index, regimes, colors=C.REGIME_COLORS, alpha=0.15):
    for regime, color in colors.items():
        mask = (regimes == regime).values
        for t in range(len(mask) - 1):
            if mask[t]:
                ax.axvspan(index[t], index[t + 1], color=color, alpha=alpha, lw=0)


def _year_axis(ax, index):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(index.min(), index.max())


# ---------------------------------------------------------------------------
# Figure 1 — raw return series with crisis shading
# ---------------------------------------------------------------------------
def fig_return_series(R_f, R_bf, G_b, window=C.WINDOW, name="fig01_return_series.pdf"):
    set_style()
    idx = R_f.index
    fig, (axf, axb, axe, axr) = plt.subplots(4, 1, figsize=(14, 16), sharex=True,
                                             gridspec_kw={"hspace": 0.12})
    for ax, mat in [(axf, R_f), (axb, R_bf)]:
        for fund in mat.columns:
            p = fund.split("_")[0]
            ma = mat[fund].reindex(idx).rolling(window).mean()
            ax.plot(ma.index, ma.values, color=C.PROVIDER_COLORS.get(p, "gray"),
                    lw=0.6, alpha=0.5)
        ax.axhline(0, color="black", lw=0.5)
    axf.set_ylabel(r"$r^{(f)}_{i,t}$ (60-day MA)")
    axb.set_ylabel(r"$r^{(b,f)}_{i,t}$ (60-day MA)")
    axf.legend(handles=[Line2D([0], [0], color=c, lw=1.5, label=p)
                        for p, c in C.PROVIDER_COLORS.items()],
               fontsize=7, ncol=5, loc="upper right")
    for col, (color, ls, lw, lab) in C.EQUITY_SERIES.items():
        ma = G_b[col].reindex(idx).rolling(window).mean()
        axe.plot(ma.index, ma.values, color=color, ls=ls, lw=lw, label=lab)
    axe.axhline(0, color="black", lw=0.5); axe.set_ylabel("Global equity\n(60-day MA)")
    axe.legend(fontsize=8, ncol=3, loc="upper right")
    for col, (color, ls, lw, lab) in C.RATE_SERIES.items():
        axr.plot(G_b[col].index, G_b[col].values, color=color, ls=ls, lw=lw, label=lab)
    axr.axhline(0, color="black", lw=0.6)
    axr.set_ylabel("Yield shocks\n(bp)"); axr.set_xlabel("Date")
    axr.legend(fontsize=8, ncol=2, loc="upper right")
    for ax in (axf, axb, axe, axr):
        shade_events(ax)
    label_events(axr)
    _year_axis(axr, idx)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 2 — AR_t with HMM regimes + benchmarks
# ---------------------------------------------------------------------------
def fig_ar_benchmarks(ar_clean, regime_series, tau, R_bf, G_b, N=C.N,
                      crisis_days=None, total_days=None,
                      name="fig02_ar_benchmarks.pdf"):
    set_style()
    idx = ar_clean.index
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[2.5, 2, 1.8, 1.8],
                           hspace=0.25)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i]) for i in range(4))

    ax1.plot(idx, ar_clean.values, color="black", lw=0.9, label=r"$AR_t$", zorder=5)
    regime_shading(ax1, idx, regime_series, alpha=0.20)
    ax1.axhline(tau, color="darkred", ls="--", lw=1.2, label=fr"$\tau={tau:.3f}$")
    ax1.axhline(1 / N, color="gray", ls=":", lw=0.8, label=fr"$1/N={1/N:.3f}$")
    ax1.set_ylabel(r"$AR_t$")
    if crisis_days is not None:
        ax1.annotate(f"Crisis days: {crisis_days}/{total_days} "
                     f"({100 * crisis_days / total_days:.1f}%)",
                     xy=(0.98, 0.05), xycoords="axes fraction", ha="right",
                     fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                           facecolor="white", alpha=0.8))
    patches = [mpatches.Patch(color=c, alpha=0.4, label=r)
               for r, c in C.REGIME_COLORS.items()]
    h, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=h + patches, fontsize=8, ncol=6, loc="upper right")

    for fund in R_bf.columns:
        ma = R_bf[fund].reindex(idx).rolling(60).mean()
        ax2.plot(ma.index, ma.values, lw=0.7, alpha=0.5,
                 color=C.PROVIDER_COLORS.get(fund.split("_")[0], "gray"))
    ax2.axhline(0, color="black", lw=0.5); ax2.set_ylabel(r"$r^{(b,f)}_{i,t}$")
    ax2.legend(handles=[Line2D([0], [0], color=c, lw=1.5, label=p)
                        for p, c in C.PROVIDER_COLORS.items()],
               fontsize=8, ncol=5, loc="upper right")

    for col, (color, ls, lw, lab) in C.EQUITY_SERIES.items():
        ma = G_b[col].reindex(idx).rolling(60).mean()
        ax3.plot(ma.index, ma.values, color=color, ls=ls, lw=1.5, label=lab)
    ax3.axhline(0, color="black", lw=0.5); ax3.set_ylabel("Equity\n(60-day MA)")
    ax3.legend(fontsize=9, ncol=3, loc="upper right")

    for col, (color, ls, lw, lab) in C.RATE_SERIES.items():
        ma = G_b[col].reindex(idx).rolling(60).mean()
        ax4.plot(ma.index, ma.values, color=color, ls=ls, lw=1.8, label=lab)
    ax4.axhline(0, color="black", lw=0.5)
    ax4.set_ylabel("Yield changes\n(60-day MA)"); ax4.set_xlabel("Date")
    ax4.legend(fontsize=9, ncol=2, loc="upper right")

    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelbottom=False)
    _year_axis(ax4, idx)
    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(idx.min(), idx.max())
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 3 — baseline vs augmented AR (levels + standardised)
# ---------------------------------------------------------------------------
def fig_baseline_vs_augmented(AR_corr, AR_augmented, name="fig03_ar_baseline_vs_augmented.pdf"):
    set_style()
    idx = AR_corr.index.intersection(AR_augmented.index)
    b, a = AR_corr.loc[idx], AR_augmented.loc[idx]
    rho = np.corrcoef(b, a)[0, 1]
    bz, az = (b - b.mean()) / b.std(), (a - a.mean()) / a.std()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 4.6), sharex=True)
    axL.plot(b.index, b, lw=1.3, label="Baseline AR ($N=40$)")
    axL.plot(a.index, a, lw=1.3, label="Augmented AR ($2N+M=85$)")
    axL.set_ylabel("Absorption ratio (correlation)"); axL.set_xlabel("Date")
    axR.plot(bz.index, bz, lw=1.3, label="Baseline AR (z)")
    axR.plot(az.index, az, lw=1.3, label=f"Augmented AR (z), $\\rho={rho:.2f}$")
    axR.axhline(0, color="black", lw=0.5)
    axR.set_ylabel("Standardised AR ($z$)"); axR.set_xlabel("Date")
    for ax in (axL, axR):
        shade_events(ax); label_events(ax)
        ax.legend(loc="upper right", fontsize=9, ncol=1)
        _year_axis(ax, idx)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figures 4/5 — dendrogram + cluster heatmap
# ---------------------------------------------------------------------------
def fig_dendrogram(cluster_res, name="fig04_dendrogram.pdf"):
    set_style()
    Z, best_k = cluster_res.linkage, cluster_res.best_k
    labels = [f"P{f.split('_')[0].split()[-1]}/{f.split('_')[1]}"
              for f in cluster_res.fund_names]
    cut = Z[-(best_k - 1), 2]
    fig, ax = plt.subplots(figsize=(14, 7))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=9,
               color_threshold=cut, above_threshold_color="gray", ax=ax)
    ax.axhline(cut, color="darkred", ls="--", lw=1.2,
               label=fr"Cut: $K^*={best_k}$ clusters")
    ax.set_ylabel("DTW distance"); ax.set_xlabel("Pension fund"); ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, name)


def fig_cluster_heatmap(cluster_res, which="dtw", name="fig05_cluster_heatmap.pdf"):
    set_style()
    from .clustering import cluster_frame
    df = cluster_frame(cluster_res, which if cluster_res.labels_dtw is not None else "hier")
    best_k = cluster_res.best_k
    num = df.pivot(index="Provider", columns="AgeCohort", values="Cluster")
    lab = "C" + df["Cluster"].astype(str)
    labels = df.assign(L=lab).pivot(index="Provider", columns="AgeCohort", values="L")
    cmap = ListedColormap(sns.color_palette("coolwarm", best_k))
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(num, cmap=cmap, annot=labels, fmt="", linewidths=1,
                linecolor="white", cbar=True, ax=ax, vmin=-0.5, vmax=best_k - 0.5)
    ax.set_xlabel("Age cohort"); ax.set_ylabel("Provider")
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 6 — emission distributions with thresholds
# ---------------------------------------------------------------------------
def fig_emission_distributions(res, name="fig06_emission_distributions.pdf"):
    set_style()
    mu, sd = res["mu"], res["sigma"]
    tau, tau_1 = res["tau"], res["tau_1"]
    lo = float(min(mu) - 4 * max(sd))
    hi = float(max(mu) + 4 * max(sd))
    ar_range = np.linspace(lo, hi, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, regime in enumerate(C.REGIME_NAMES):
        color = C.REGIME_COLORS[regime]
        pdf = scipy_norm.pdf(ar_range, mu[i], sd[i])
        ax.plot(ar_range, pdf, color=color, lw=2.0,
                label=f"{regime} ($\\hat\\mu={mu[i]:.3f}$, $\\hat\\sigma={sd[i]:.3f}$)")
        ax.fill_between(ar_range, pdf, alpha=0.20, color=color)
        ax.axvline(mu[i], color=color, lw=1.0, ls="--", alpha=0.8)
    ax.axvline(tau, color="darkred", lw=1.5, ls="--",
               label=fr"Crisis threshold $\tau={tau:.2f}$")
    ax.axvline(tau_1, color="gray", lw=1.0, ls=":",
               label=fr"Low$|$Moderate $\tau_1={tau_1:.2f}$")
    ax.set_xlabel(r"Absorption ratio $AR_t$"); ax.set_ylabel("Density")
    ax.set_xlim(lo, hi); ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 7 — scatter of AR_t vs interpretive series, coloured by regime
# ---------------------------------------------------------------------------
def fig_scatter_correlations(interp_df, tau, name="fig07_scatter_correlations.pdf"):
    set_style()
    from scipy import stats as ss
    panels = [("MAAR", "Mean Absolute Active Return"),
              ("MSCI_World", "MSCI World"), ("MSCI_Europe", "MSCI Europe"),
              ("SP500", "S&P 500"), ("dY_10Y", "10Y Yield Change"),
              ("dY_2Y", "2Y Yield Change")]
    fig, axes = plt.subplots(3, 2, figsize=(14, 15)); axes = axes.flatten()
    for ax, (col, title) in zip(axes, panels):
        for regime in C.REGIME_NAMES:
            sub = interp_df[interp_df["Regime"] == regime]
            ax.scatter(sub["AR"], sub[col], c=C.REGIME_COLORS[regime], s=14,
                       alpha=0.55, edgecolors="none", label=regime)
        x, y = interp_df["AR"].values, interp_df[col].values
        sl, ic, *_ = ss.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, ic + sl * xl, color="black", lw=1.8, zorder=10)
        r_p, p_p = ss.pearsonr(x, y)
        ax.annotate(f"Pearson r = {r_p:.3f}", xy=(0.03, 0.97),
                    xycoords="axes fraction", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))
        ax.axvline(tau, color="darkred", lw=1.0, ls="--", alpha=0.5)
        ax.set_xlabel(r"$AR_t$"); ax.set_title(title, fontsize=12, fontweight="bold")
    handles = [plt.scatter([], [], c=C.REGIME_COLORS[r], s=40, label=r)
               for r in C.REGIME_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 8 — cluster MAAR amplification (normal vs crisis)
# ---------------------------------------------------------------------------
def fig_cluster_maar(maar_df, name="fig08_cluster_regime_MAAR.pdf"):
    set_style()
    extreme, middle = "#2166ac", "#d73027"
    colors = [extreme if "Extreme" in maar_df.loc[c, "Cohort_type"] else middle
              for c in maar_df.index]
    x = np.arange(len(maar_df)); width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.35})
    ax1.bar(x - width / 2, maar_df["MAAR_normal"] * 1000, width, color=colors,
            alpha=0.35, edgecolor="black", lw=0.6)
    ax1.bar(x + width / 2, maar_df["MAAR_crisis"] * 1000, width, color=colors,
            alpha=0.90, edgecolor="black", lw=0.6)
    for i, (_c, row) in enumerate(maar_df.iterrows()):
        ax1.text(i + width / 2, row["MAAR_crisis"] * 1000 + 0.02,
                 f"{row['Ratio']:.2f}x", ha="center", va="bottom",
                 fontsize=7.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n(n={int(maar_df.loc[c, 'N_funds'])})"
                         for c in maar_df.index], fontsize=9)
    ax1.set_ylabel("Mean absolute active return\n(x1000), crisis vs normal", fontsize=9)
    ax1.set_xlabel("Clusters")
    ax2.barh(maar_df.index[::-1], maar_df["Ratio"].values[::-1],
             color=colors[::-1], alpha=0.8, edgecolor="black", lw=0.6)
    ax2.axvline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax2.set_xlabel("Crisis / normal ratio")
    valid = maar_df["Ratio"].replace([np.inf, -np.inf], np.nan)
    ax2.set_xlim(0, valid.max() + 0.15)
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figures 9/10 — regime classification + transition matrix
# ---------------------------------------------------------------------------
def fig_regime_classification(res, name="fig09_regime_classification.pdf"):
    set_style()
    series, labels = res["ar_clean"], res["regime_series"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(series.index, series, color="black", lw=0.8)
    for regime in C.REGIME_NAMES:
        mask = labels == regime
        ax.scatter(series.index[mask], series[mask], s=8,
                   c=C.REGIME_COLORS[regime], label=regime)
    ax.legend(ncol=3); ax.set_title("HMM Regime Classification")
    fig.tight_layout()
    return _save(fig, name)


def fig_transition_matrix(em_model, name="fig10_transition_matrix.pdf"):
    set_style()
    order = np.argsort(em_model.means_.flatten())
    A = em_model.transmat_[np.ix_(order, order)]
    labels = ["Low (0)", "Moderate (1)", "High (2)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(A, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Transition Matrix"); ax.set_xlabel("To"); ax.set_ylabel("From")
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 11 — overlap ACF check
# ---------------------------------------------------------------------------
def fig_acf(acf_df, window=C.WINDOW, name="fig11_overlap_acf.pdf"):
    set_style()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acf_df["lag"], acf_df["empirical"], label="Empirical AR autocorrelation")
    ax.plot(acf_df["lag"], acf_df["overlap_benchmark"], "--",
            label=f"Overlap-only benchmark (decays by lag {window})")
    ax.axvline(window, color="grey", ls=":", label=f"window = {window}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation"); ax.legend()
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 12 — covariance-estimator comparison
# ---------------------------------------------------------------------------
def fig_estimator_comparison(comparison, name="fig12_covariance_comparison.pdf"):
    set_style()
    styles = {"sample": ("black", 0.9), "ledoit_wolf": ("#2166ac", 0.7),
              "mcd": ("#d73027", 0.7), "mp_filtered": ("#1a9850", 0.7)}
    fig, ax = plt.subplots(figsize=(14, 5))
    for col, (color, lw) in styles.items():
        if col in comparison:
            ax.plot(comparison.index, comparison[col], color=color, lw=lw,
                    alpha=0.85, label=col.replace("_", " ").title())
    ax.set_ylabel(r"$AR_t$"); ax.set_xlabel("Date")
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    _year_axis(ax, comparison.index)
    return _save(fig, name)
