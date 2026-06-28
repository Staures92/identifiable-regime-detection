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
    """AR_t + fund benchmarks + equity factors + yield shocks (4 stacked panels).

    Mirrors the manuscript figure: panel 1 is AR_t alone with crisis shading;
    panels 2-3 are 60-day moving averages; panel 4 shows the *raw* yield shocks
    in basis points. ``regime_series``/``tau`` are accepted for call-site
    compatibility but not drawn here (regime overlays live in
    ``fig_regime_classification``)."""
    set_style()
    idx = ar_clean.index
    fig = plt.figure(figsize=(14, 14))
    gs = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[2.5, 2, 2, 2],
                           hspace=0.20)
    ax1, ax2, ax3, ax4 = (fig.add_subplot(gs[i]) for i in range(4))
    axes = [ax1, ax2, ax3, ax4]

    # Panel 1 — absorption ratio
    ax1.plot(idx, ar_clean.values, color="black", lw=1.0, label=r"$AR_t$", zorder=5)
    ax1.set_ylabel(r"$AR_t$")
    ax1.set_ylim(ar_clean.min() * 0.97, ar_clean.max() * 1.02)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Panel 2 — fund-specific benchmarks (60-day MA)
    for fund in R_bf.columns:
        color = C.PROVIDER_COLORS.get(fund.split("_")[0], "gray")
        ma = R_bf[fund].reindex(idx).rolling(60).mean()
        ax2.plot(ma.index, ma.values, color=color, lw=0.7, alpha=0.50)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("$r^{(b,f)}_{i,t}$\n(60-day MA)")
    ax2.legend(handles=[Line2D([0], [0], color=c, lw=1.5, label=p)
                        for p, c in C.PROVIDER_COLORS.items()],
               fontsize=8, loc="upper right", framealpha=0.9, ncol=5)

    # Panel 3 — global equity factors (60-day MA)
    for col, (color, ls, lw, lab) in C.EQUITY_SERIES.items():
        ma = G_b[col].reindex(idx).rolling(60).mean()
        ax3.plot(ma.index, ma.values, color=color, ls=ls, lw=lw, alpha=0.90, label=lab)
    ax3.axhline(0, color="black", lw=0.5)
    ax3.set_ylabel("Global equity\nfactors\n(60-day MA)")
    ax3.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=3)

    # Panel 4 — euro-area yield shocks (raw, bp)
    for col, (color, ls, lw, lab) in C.RATE_SERIES.items():
        s = G_b[col].reindex(idx)
        ax4.plot(s.index, s.values, color=color, ls=ls, lw=lw, alpha=0.85, label=lab)
    ax4.axhline(0, color="black", lw=0.6)
    ax4.set_ylabel("Yield shocks\n(bp)"); ax4.set_xlabel("Date")
    ax4.legend(fontsize=8, loc="upper right", framealpha=0.9, ncol=2)

    for ax in axes:
        shade_events(ax)
    label_events(ax4)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelbottom=False)
    _year_axis(ax4, idx)
    for ax in axes:
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
        ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
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


def fig_cluster_heatmap(cluster_res, which="dtw", name="fig05_cluster_comparison.pdf"):
    """Two stacked Provider x cohort heatmaps: hierarchical DTW (panel A, 'C')
    vs DTW k-means (panel B, 'KM'), with their ARI in the panel-B title."""
    set_style()
    from sklearn.metrics import adjusted_rand_score

    best_k = cluster_res.best_k
    providers, cohorts = C.PROVIDERS, C.COHORTS
    labels_hier = np.asarray(cluster_res.labels_hier)
    labels_dtw = (np.asarray(cluster_res.labels_dtw)
                  if cluster_res.labels_dtw is not None else labels_hier)

    heat_hier = pd.DataFrame(index=providers, columns=cohorts, dtype=float)
    heat_dtw = pd.DataFrame(index=providers, columns=cohorts, dtype=float)
    for i, fund in enumerate(cluster_res.fund_names):
        provider, cohort = fund.split("_")
        heat_hier.loc[provider, cohort] = labels_hier[i]
        heat_dtw.loc[provider, cohort] = labels_dtw[i]

    ari = adjusted_rand_score(labels_hier, labels_dtw)
    colors = sns.color_palette("tab10", best_k)
    cmap = ListedColormap(colors)
    # hierarchical labels are 1..K, k-means labels are 0..K-1
    norm_hier = BoundaryNorm(np.arange(0.5, best_k + 1.5), best_k)
    norm_dtw = BoundaryNorm(np.arange(-0.5, best_k + 0.5), best_k)

    fig, axes = plt.subplots(2, 1, figsize=(6, 10), sharey=True)
    for ax, heat, norm, prefix, title in [
        (axes[0], heat_hier, norm_hier, "C", f"(A) Hierarchical DTW (K={best_k})"),
        (axes[1], heat_dtw, norm_dtw, "KM",
         f"(B) DTW k-means (K={best_k})\nARI = {ari:.3f}")]:
        ax.imshow(heat.values, cmap=cmap, norm=norm, aspect="auto")
        for r in range(heat.shape[0]):
            for cc in range(heat.shape[1]):
                val = heat.iloc[r, cc]
                if not np.isnan(val):
                    ax.text(cc, r, f"{prefix}{int(val)}", ha="center", va="center",
                            fontsize=7, fontweight="bold")
        ax.set_title(title, fontsize=8, fontweight="bold")

    xticklabels = [f"{c}\n({C.COHORT_BANDS[c]})" for c in cohorts]
    for ax in axes:
        ax.set_xticks(range(len(cohorts))); ax.set_xticklabels(xticklabels, fontsize=7)
        ax.set_yticks(range(len(providers))); ax.set_yticklabels(providers, fontsize=7)
        ax.set_xlabel("Age cohort")
        ax.set_xticks(np.arange(-0.5, len(cohorts)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(providers)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)
    axes[0].set_ylabel("Provider", fontsize=7)

    axes[0].legend(handles=[Patch(facecolor=colors[i - 1], label=f"C{i}")
                            for i in range(1, best_k + 1)],
                   title="Hierarchical", loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=6, framealpha=0.9)
    axes[1].legend(handles=[Patch(facecolor=colors[i], label=f"KM{i}")
                            for i in range(best_k)],
                   title="DTW k-means", loc="upper center",
                   bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=6, framealpha=0.9)
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
    """AR_t vs each interpretive series, coloured by regime, with a single OLS
    fit and Pearson/Spearman/Kendall stats; crisis threshold marked."""
    set_style()
    from scipy import stats as ss

    panels = [("MAAR", "Mean Absolute Active Return", r"$|r^{(f)}_{i,t}-r^{(b,f)}_{i,t}|$"),
              ("MSCI_World", "MSCI World", r"$r^{(b,g)}_{\mathrm{MSCI\,World},t}$"),
              ("MSCI_Europe", "MSCI Europe", r"$r^{(b,g)}_{\mathrm{MSCI\,Europe},t}$"),
              ("SP500", "S&P 500", r"$r^{(b,g)}_{\mathrm{SP500},t}$"),
              ("dY_10Y", "10-Year Yield Change", r"$\Delta Y^{(10Y)}_t$"),
              ("dY_2Y", "2-Year Yield Change", r"$\Delta Y^{(2Y)}_t$")]

    def sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

    ar = interp_df["AR"].values
    pad = (ar.max() - ar.min()) * 0.02
    xlim = (ar.min() - pad, ar.max() + pad)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15)); axes = axes.flatten()
    for ax, (col, title, ylabel) in zip(axes, panels):
        for regime in C.REGIME_NAMES:
            sub = interp_df[interp_df["Regime"] == regime]
            ax.scatter(sub["AR"], sub[col], c=C.REGIME_COLORS[regime], s=14,
                       alpha=0.55, edgecolors="none", label=regime,
                       zorder=2 + C.REGIME_NAMES.index(regime))
        x, y = interp_df["AR"].values, interp_df[col].values
        slope, intercept, *_ = ss.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, intercept + slope * xl, color="black", lw=1.8, zorder=10)

        r_p, p_p = ss.pearsonr(x, y)
        r_s, p_s = ss.spearmanr(x, y)
        r_k, p_k = ss.kendalltau(x, y)
        txt = (f"Pearson r = {r_p:.3f} ({sig(p_p)})\n"
               f"Spearman \u03c1 = {r_s:.3f} ({sig(p_s)})\n"
               f"Kendall \u03c4 = {r_k:.3f} ({sig(p_k)})")
        xy, ha = ((0.03, 0.97), "left") if slope >= 0 else ((0.97, 0.97), "right")
        ax.annotate(txt, xy=xy, xycoords="axes fraction", ha=ha, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="gray", alpha=0.9))
        ax.axvline(tau, color="darkred", lw=1.0, ls="--", alpha=0.5, zorder=1)
        ax.set_xlim(xlim); ax.set_xlabel(r"$AR_t$"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")

    handles = [plt.scatter([], [], c=C.REGIME_COLORS[r], s=40, label=r)
               for r in C.REGIME_NAMES]
    handles.append(Line2D([0], [0], color="black", lw=2.0, label="OLS regression"))
    handles.append(Line2D([0], [0], color="darkred", ls="--", lw=2.0, alpha=0.5,
                          label=fr"Crisis threshold $\tau={tau:.3f}$"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 7 — regime-conditional scatter (two-slope fits)
# ---------------------------------------------------------------------------
def fig_regime_conditional_scatter(interp_df, fitted_models, tau,
                                   name="fig08_regime_conditional_scatter.pdf"):
    """AR_t vs interpretive series with the two fitted lines from the
    regime-conditional OLS: a calm fit (slope b1) and a High-Concentration fit
    (slope b1+b3). The visual gap between the slopes *is* b3."""
    set_style()
    panel_info = {
        "MAAR": ("Mean Absolute Active Return", r"$|r^{(f)}_{i,t} - r^{(b,f)}_{i,t}|$"),
        "MSCI_World": ("MSCI World", r"$r^{(b,g)}_{\mathrm{MSCI\,World},t}$"),
        "MSCI_Europe": ("MSCI Europe", r"$r^{(b,g)}_{\mathrm{MSCI\,Europe},t}$"),
        "SP500": ("S&P 500", r"$r^{(b,g)}_{\mathrm{SP500},t}$"),
        "dY_10Y": ("10-Year Yield Change", r"$\Delta Y^{(10Y)}_t$"),
        "dY_2Y": ("2-Year Yield Change", r"$\Delta Y^{(2Y)}_t$"),
    }
    order = ["MAAR", "MSCI_World", "MSCI_Europe", "SP500", "dY_10Y", "dY_2Y"]
    ar = interp_df["AR"].values

    fig, axes = plt.subplots(3, 2, figsize=(13, 10)); axes = axes.flatten()
    for ax, col in zip(axes, order):
        title, ylabel = panel_info[col]
        y = interp_df[col].values
        b0, b1, b2, b3 = fitted_models[col].params

        for regime in C.REGIME_NAMES:
            mask = (interp_df["Regime"] == regime).values
            ax.scatter(ar[mask], y[mask], c=C.REGIME_COLORS[regime], s=14,
                       alpha=0.55, edgecolors="none", label=regime)

        calm = (interp_df["Regime"] != "High-Concentration").values
        if calm.any():
            grid = np.linspace(ar[calm].min(), ar[calm].max(), 100)
            ax.plot(grid, b0 + b1 * grid, color="#1F4E79", lw=2.2,
                    label=f"Calm fit (slope $b_1$={b1:.3f})", zorder=5)
        high = (interp_df["Regime"] == "High-Concentration").values
        if high.any():
            grid = np.linspace(ar[high].min(), ar[high].max(), 100)
            ax.plot(grid, (b0 + b2) + (b1 + b3) * grid, color="#8B0000", lw=2.2,
                    label=f"High-Risk fit (slope $b_1+b_3$={b1 + b3:.3f})", zorder=5)

        ax.axvline(tau, color="gray", ls=":", lw=1.0, alpha=0.7)
        ax.set_xlabel(r"$AR_t$", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=28)
        ax.grid(True, alpha=0.25, ls="--")
        handles, labs = ax.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(handles, labs) if "fit" in l.lower()]
        ax.legend([h for h, _ in keep], [l for _, l in keep], loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, framealpha=0.9,
                  borderaxespad=0.3)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------------------
# Figure 8 — cluster MAAR amplification (normal vs crisis)
# ---------------------------------------------------------------------------
def fig_cluster_maar(maar_df, name="fig09_cluster_regime_MAAR.pdf"):
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
def fig_regime_classification(res, name="fig10_regime_classification.pdf"):
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


def fig_transition_matrix(em_model, name="fig11_transition_matrix.pdf"):
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
def fig_acf(acf_df, window=C.WINDOW, name="fig12_overlap_acf.pdf"):
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
def fig_estimator_comparison(comparison, name="fig13_covariance_comparison.pdf"):
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
