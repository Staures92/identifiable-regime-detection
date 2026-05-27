"""
All publication figures, numbered to match the paper.

Figure index
------------
fig01_return_series                 Returns: R^(f), R^(b,f), R^(b,g)
fig02_ar_benchmarks_events          AR_t with event shading
fig03_ar_benchmarks_regimes         AR_t with HMM regime shading
fig04_ar_baseline_vs_augmented      Baseline vs augmented AR_t
fig05_dendrogram                    Ward hierarchical clustering dendrogram
fig06_cluster_heatmap               Cluster heatmap (provider x age cohort)
fig07_emission_distributions        HMM emission densities + thresholds
fig08_cluster_regime_tracking_err   Cluster x regime tracking error
fig09_scatter_correlations          AR_t vs benchmarks (scatter + OLS)
fig10_regime_conditional_scatter    Regime-conditional regression
fig11_risk_decomposition            Systematic vs specific risk
fig12_covariance_comparison         Sample vs Ledoit-Wolf vs MCD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats
from scipy.stats import norm as scipy_norm
from scipy.cluster.hierarchy import dendrogram, fcluster

from .config import (
    AGE_GROUPS,
    EVENT_WINDOWS,
    FIGURES_DIR,
    GLOBAL_BENCHMARK_STYLES,
    PLT_RCPARAMS,
    PROVIDER_COLORS,
    PROVIDERS,
    REGIME_COLORS,
)


# =========================================================
# COMMON STYLING
# =========================================================
def apply_style():
    plt.rcParams.update(PLT_RCPARAMS)


def _save(fig, name):
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def _shade_events(axes):
    """Shade COVID-19 and 2022 rate-hike windows on a list of axes."""
    for _, (start, end, color, alpha) in EVENT_WINDOWS.items():
        for ax in axes:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=alpha, color=color, linewidth=0)


def _shade_regimes(axes, regime_series, ar_index, alpha_main=0.20,
                   alpha_panel=0.07):
    """Shade HMM regimes on a list of axes."""
    for regime, color in REGIME_COLORS.items():
        mask = (regime_series == regime).values
        for t in range(len(mask) - 1):
            if mask[t]:
                for i, ax in enumerate(axes):
                    a = alpha_main if i == 0 else alpha_panel
                    ax.axvspan(ar_index[t], ar_index[t + 1],
                               alpha=a, color=color, linewidth=0)


# =========================================================
# fig01 — Return series (3 panels)
# =========================================================
def fig01_return_series(R_f, R_bf, R_bg, ar_index, name="fig01_return_series"):
    apply_style()
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 15), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    ax_f, ax_bf, ax_bg = axes
    MA_WINDOW = 60

    cohort_style = {
        "AG1": dict(linewidth=1.2, linestyle="-",  alpha=0.9),
        "AG8": dict(linewidth=1.2, linestyle="--", alpha=0.9),
    }
    for ag in ["AG2", "AG3", "AG4", "AG5", "AG6", "AG7"]:
        cohort_style[ag] = dict(linewidth=0.6, linestyle="-", alpha=0.5)

    provider_step = 0.006
    ag_step       = 0.0005

    def panel(ax, data, ylabel):
        ytick_pos, ytick_lab = [], []
        seen = set()
        for p_idx, provider in enumerate(PROVIDERS):
            p_color  = PROVIDER_COLORS[provider]
            p_offset = p_idx * provider_step
            for ag_idx, ag in enumerate(AGE_GROUPS):
                fund = f"{provider}_{ag}"
                if fund not in data.columns:
                    continue
                style  = cohort_style[ag]
                offset = p_offset + ag_idx * ag_step
                series = data[fund].reindex(ar_index)
                ma     = series.rolling(MA_WINDOW).mean()
                if provider not in seen:
                    seen.add(provider)
                    ytick_pos.append(p_offset)
                    ytick_lab.append(provider.replace("Provider ", "P"))
                ax.plot(ma.index, ma.values + offset,
                        color=p_color, **style)
            ax.axhline(p_offset, color=p_color, linewidth=0.3,
                       alpha=0.3, zorder=0)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lab, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)

    panel(ax_f,  R_f,  "$r^{(f)}_{i,t}$ (60-day MA)\noffset by provider")
    panel(ax_bf, R_bf, "$r^{(b,f)}_{i,t}$ (60-day MA)\noffset by provider")

    # Panel 3: global benchmarks
    for col, (color, ls, lw, label) in GLOBAL_BENCHMARK_STYLES.items():
        if col not in R_bg.columns:
            continue
        ma = R_bg[col].reindex(ar_index).rolling(MA_WINDOW).mean()
        ax_bg.plot(ma.index, ma.values, color=color, linewidth=lw,
                   linestyle=ls, alpha=0.9, label=label)
    ax_bg.axhline(0, color="black", linewidth=0.5)
    ax_bg.set_ylabel("$r^{(b,g)}_{m,t}$\n(60-day MA)", fontsize=11)
    ax_bg.set_xlabel("Date", fontsize=11)
    ax_bg.legend(fontsize=9, loc="upper right", ncol=3)

    _shade_events(axes)

    for ax in [ax_f, ax_bf]:
        ax.tick_params(labelbottom=False)
    ax_bg.xaxis.set_major_locator(mdates.YearLocator())
    ax_bg.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        ax.set_xlim(ar_index.min(), ar_index.max())

    return _save(fig, name)


# =========================================================
# fig02 / fig03 — AR_t with shading
# =========================================================
def _ar_benchmarks_figure(AR_clean, R_bf, R_bg,
                          shading_mode, regime_series=None, tau=None,
                          name="fig"):
    """Shared core for fig02 (events) and fig03 (regimes)."""
    apply_style()
    fig = plt.figure(figsize=(14, 12))
    gs  = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[2.5, 2, 2],
                            hspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ax1.plot(AR_clean.index, AR_clean.values,
             color="black", linewidth=0.9, zorder=3, label="$AR_t$")
    ax1.set_ylabel("$AR_t$", fontsize=11)
    ax1.set_ylim(0.68, 0.97)

    if shading_mode == "events":
        _shade_events([ax1, ax2, ax3])
    elif shading_mode == "regimes":
        _shade_regimes([ax1, ax2, ax3], regime_series, AR_clean.index)
        if tau is not None:
            ax1.axhline(tau, color="darkred", linewidth=1.2,
                        linestyle="--", zorder=4,
                        label=f"$\\tau = {tau:.3f}$")
        regime_patches = [mpatches.Patch(color=c, alpha=0.4, label=r)
                          for r, c in REGIME_COLORS.items()]
        h_ar, _ = ax1.get_legend_handles_labels()
        ax1.legend(handles=h_ar + regime_patches, loc="upper right",
                   fontsize=8.5, framealpha=0.9, ncol=6)

    # Panel 2: fund-specific benchmarks
    seen = set()
    for fund in R_bf.columns:
        provider = fund.split("_")[0]
        color    = PROVIDER_COLORS.get(provider, "gray")
        label    = provider if provider not in seen else "_nolegend_"
        seen.add(provider)
        ma = R_bf[fund].reindex(AR_clean.index).rolling(60).mean()
        ax2.plot(ma.index, ma.values, color=color, linewidth=0.7,
                 alpha=0.5, label=label)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("$r^{(b,f)}_{i,t}$\n(60-day MA)", fontsize=11)
    ax2.set_ylim(-0.008, 0.008)
    ax2.legend(fontsize=8.5, loc="upper right", ncol=5)

    # Panel 3: global benchmarks
    for col, (color, ls, lw, label) in GLOBAL_BENCHMARK_STYLES.items():
        if col not in R_bg.columns:
            continue
        ma = R_bg[col].reindex(AR_clean.index).rolling(60).mean()
        ax3.plot(ma.index, ma.values, color=color, linewidth=lw,
                 linestyle=ls, alpha=0.9, label=label, zorder=5)
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylabel("$r^{(b,g)}_{m,t}$\n(60-day MA)", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.legend(fontsize=9, loc="upper right", ncol=5)

    for ax in [ax1, ax2]:
        ax.tick_params(labelbottom=False)
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(AR_clean.index.min(), AR_clean.index.max())

    return _save(fig, name)


def fig02_ar_benchmarks_events(AR_clean, R_bf, R_bg):
    return _ar_benchmarks_figure(
        AR_clean, R_bf, R_bg, shading_mode="events",
        name="fig02_ar_benchmarks_events",
    )


def fig03_ar_benchmarks_regimes(AR_clean, R_bf, R_bg, regime_series, tau):
    return _ar_benchmarks_figure(
        AR_clean, R_bf, R_bg, shading_mode="regimes",
        regime_series=regime_series, tau=tau,
        name="fig03_ar_benchmarks_regimes",
    )


# =========================================================
# fig04 — Baseline vs augmented AR
# =========================================================
def fig04_ar_baseline_vs_augmented(AR_baseline, AR_augmented,
                                   name="fig04_ar_baseline_vs_augmented"):
    apply_style()
    common = AR_baseline.index.intersection(AR_augmented.index)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(common, AR_baseline.loc[common], label="Baseline AR", alpha=0.8)
    ax.plot(common, AR_augmented.loc[common], label="Augmented AR", alpha=0.8)
    ax.set_ylabel("Absorption Ratio")
    ax.legend()
    fig.tight_layout()
    return _save(fig, name)


# =========================================================
# fig05 — Dendrogram
# =========================================================
def fig05_dendrogram(Z_linkage, fund_names, k_star=8, name="fig05_dendrogram"):
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    cut_heights = [Z_linkage[i, 2] for i in range(len(Z_linkage))
                   if len(set(fcluster(Z_linkage, Z_linkage[i, 2],
                                       criterion="distance"))) <= k_star]
    cut_height = min(cut_heights) if cut_heights else Z_linkage[-1, 2] * 0.5

    clean_labels = [f.replace("Provider ", "P").replace("_AG", "/AG")
                    for f in fund_names]
    dendrogram(Z_linkage, labels=clean_labels, leaf_rotation=90,
               leaf_font_size=8, color_threshold=cut_height,
               above_threshold_color="gray", ax=ax)

    ax.axhline(cut_height, color="darkred", linewidth=1.2, linestyle="--",
               label=f"Cut: $K_c^* = {k_star}$ clusters")
    ax.set_ylabel("Ward linkage distance", fontsize=11)
    ax.set_xlabel("Pension fund", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, name)


# =========================================================
# fig06 — Cluster heatmap
# =========================================================
def fig06_cluster_heatmap(cluster_df_w, k_star=8,
                          name="fig06_cluster_heatmap"):
    apply_style()
    cluster_matrix = pd.DataFrame(
        index=PROVIDERS, columns=AGE_GROUPS, dtype=float,
    )
    for _, row in cluster_df_w.iterrows():
        provider, ag = row["Fund"].split("_")
        cluster_matrix.loc[provider, ag] = row["Cluster"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_k = [
        "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
        "#fddbc7", "#f4a582", "#d6604d", "#b2182b",
    ][:k_star]
    cmap = ListedColormap(colors_k)
    norm = BoundaryNorm(boundaries=np.arange(0.5, k_star + 1.5), ncolors=k_star)

    im = ax.imshow(cluster_matrix.values, cmap=cmap, norm=norm, aspect="auto")

    for i in range(len(PROVIDERS)):
        for j in range(len(AGE_GROUPS)):
            v = cluster_matrix.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"C{int(v)}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    ax.set_xticks(range(len(AGE_GROUPS)))
    ax.set_xticklabels(AGE_GROUPS, fontsize=9)
    ax.set_yticks(range(len(PROVIDERS)))
    ax.set_yticklabels(PROVIDERS, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, ticks=range(1, k_star + 1),
                        shrink=0.8, pad=0.02)
    cbar.set_label("Cluster", fontsize=10)
    cbar.set_ticklabels([f"C{i}" for i in range(1, k_star + 1)])

    ax.set_xlabel("Age cohort", fontsize=11)
    ax.set_ylabel("Provider", fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(AGE_GROUPS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(PROVIDERS)),  minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    return _save(fig, name)


# =========================================================
# fig07 — Emission distributions
# =========================================================
def fig07_emission_distributions(regime_params, tau, tau_1,
                                 name="fig07_emission_distributions"):
    """
    regime_params : dict {regime_name: (mu, sigma)} (sorted by mu).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ar_range = np.linspace(0.70, 0.97, 300)

    for regime, (mu, sigma) in regime_params.items():
        color = REGIME_COLORS[regime]
        pdf   = scipy_norm.pdf(ar_range, mu, sigma)
        ax.plot(ar_range, pdf, color=color, linewidth=2.0,
                label=f"{regime} ($\\hat{{\\mu}}={mu:.3f}$, "
                      f"$\\hat{{\\sigma}}={sigma:.3f}$)")
        ax.fill_between(ar_range, pdf, alpha=0.2, color=color)
        ax.axvline(mu, color=color, linewidth=1.0, linestyle="--", alpha=0.8)

    ax.axvline(tau,   color="darkred", linewidth=1.5, linestyle="--",
               label=f"Crisis threshold $\\tau = {tau:.2f}$")
    ax.axvline(tau_1, color="gray",    linewidth=1.0, linestyle=":",
               label=f"Low$|$Moderate boundary $\\tau_1 = {tau_1:.2f}$")

    ax.set_xlabel("Absorption ratio $AR_t$", fontsize=12)
    ax.set_ylabel("Emission density", fontsize=12)
    ax.set_xlim(0.70, 0.97)
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, name)


# =========================================================
# fig08 — Cluster x regime tracking error
# =========================================================
def fig08_cluster_regime_tracking_error(results_df,
                                        name="fig08_cluster_regime_tracking_error"):
    apply_style()
    COLOR_EXTREME = "#2166ac"
    COLOR_MIDDLE  = "#d73027"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5),
        gridspec_kw={"width_ratios": [3, 1.2], "wspace": 0.35},
    )

    clusters = results_df.index.tolist()
    x        = np.arange(len(clusters))
    width    = 0.35
    bar_map  = {"Extreme (AG1+AG8)": COLOR_EXTREME,
                "Middle (AG2-AG7)":  COLOR_MIDDLE}
    colors   = [bar_map[results_df.loc[c, "Cohort_type"]] for c in clusters]

    ax1.bar(x - width/2, results_df["TE_normal"] * 1000, width,
            color=colors, alpha=0.35, edgecolor="black", linewidth=0.6)
    ax1.bar(x + width/2, results_df["TE_crisis"] * 1000, width,
            color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)

    for i, (c, row) in enumerate(results_df.iterrows()):
        ax1.text(i + width/2, row["TE_crisis"] * 1000 + 0.02,
                 f"{row['Ratio']:.2f}\u00d7",
                 ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{c}\n(n={results_df.loc[c,'N_funds']})" for c in clusters],
        fontsize=9,
    )
    ax1.set_ylabel("Mean tracking error ($\\times 10^{-3}$)", fontsize=11)
    ax1.set_xlabel("Cluster", fontsize=11)

    legend_p1 = [
        mpatches.Patch(facecolor=COLOR_EXTREME, alpha=0.35, edgecolor="black",
                       label="Extreme cohorts — normal"),
        mpatches.Patch(facecolor=COLOR_EXTREME, alpha=0.9,  edgecolor="black",
                       label="Extreme cohorts — High-Risk"),
        mpatches.Patch(facecolor=COLOR_MIDDLE,  alpha=0.35, edgecolor="black",
                       label="Middle cohorts — normal"),
        mpatches.Patch(facecolor=COLOR_MIDDLE,  alpha=0.9,  edgecolor="black",
                       label="Middle cohorts — High-Risk"),
    ]
    ax1.legend(handles=legend_p1, fontsize=8.5, loc="upper right",
               framealpha=0.9, ncol=2)

    ratio_colors = [bar_map[results_df.loc[c, "Cohort_type"]] for c in clusters]
    ax2.barh(clusters[::-1], results_df["Ratio"].values[::-1],
             color=ratio_colors[::-1], alpha=0.8,
             edgecolor="black", linewidth=0.6)
    ax2.axvline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)

    for i, (c, row) in enumerate(results_df.iloc[::-1].iterrows()):
        ax2.text(row["Ratio"] + 0.005, i, f"{row['Ratio']:.2f}\u00d7",
                 va="center", fontsize=8.5)

    ax2.set_xlabel("Crisis / normal ratio", fontsize=11)
    ax2.set_xlim(0, results_df["Ratio"].max() + 0.15)
    ax2.legend(handles=[Line2D([0], [0], color="black", linewidth=1.0,
                               linestyle="--", label="No amplification")],
               fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save(fig, name)


# =========================================================
# fig09 — Scatter plots with OLS and 3 correlations
# =========================================================
def fig09_scatter_correlations(scatter_df, tau=0.851,
                               name="fig09_scatter_correlations"):
    apply_style()
    panels = [
        ("Tracking_Error", "Average tracking error",
         r"$|r^{(f)}_{i,t} - r^{(b,f)}_{i,t}|$"),
        ("MSCI_World",     "MSCI World",
         r"$r^{(b,g)}_{\mathrm{MSCI\,World},t}$"),
        ("MSCI_Europe",    "MSCI Europe",
         r"$r^{(b,g)}_{\mathrm{MSCI\,Europe},t}$"),
        ("SP500",          "S&P 500",
         r"$r^{(b,g)}_{\mathrm{SP500},t}$"),
    ]
    regime_order = ["Low-Risk", "Moderate-Risk", "High-Risk"]
    ar_min, ar_max = scatter_df["AR"].min(), scatter_df["AR"].max()
    pad   = (ar_max - ar_min) * 0.02
    xlim  = (ar_min - pad, ar_max + pad)

    def sig(p):
        return "***" if p < 0.001 else ("**" if p < 0.01
                else ("*" if p < 0.05 else "n.s."))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for idx, (col, title, ylabel) in enumerate(panels):
        ax = axes[idx]
        for regime in regime_order:
            mask = scatter_df["Regime"] == regime
            sub  = scatter_df[mask]
            ax.scatter(sub["AR"], sub[col], c=REGIME_COLORS[regime],
                       s=14, alpha=0.55, edgecolors="none", label=regime)

        x = scatter_df["AR"].values
        y = scatter_df[col].values
        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
        xg = np.linspace(x.min(), x.max(), 100)
        ax.plot(xg, slope * xg + intercept, color="black", linewidth=1.8,
                alpha=0.9, zorder=10)

        r_p, p_p = scipy_stats.pearsonr(x, y)
        rho, p_s = scipy_stats.spearmanr(x, y)
        tau_k, p_k = scipy_stats.kendalltau(x, y)
        stats_text = (f"Pearson $r = {r_p:.3f}$ ({sig(p_p)})\n"
                      f"Spearman $\\rho = {rho:.3f}$ ({sig(p_s)})\n"
                      f"Kendall $\\tau = {tau_k:.3f}$ ({sig(p_k)})")

        xy_pos, ha = ((0.03, 0.97), "left") if slope >= 0 \
                     else ((0.97, 0.97), "right")
        ax.annotate(stats_text, xy=xy_pos, xycoords="axes fraction",
                    ha=ha, va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white", edgecolor="gray",
                              alpha=0.9))

        ax.set_xlabel("$AR_t$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(xlim)
        ax.axvline(tau, color="darkred", linewidth=0.8,
                   linestyle="--", alpha=0.5)

    handles = [plt.scatter([], [], c=REGIME_COLORS[r], s=40,
                           edgecolors="none", label=r)
               for r in regime_order]
    handles += [
        Line2D([0], [0], color="black",   linewidth=2.2,
               label="OLS regression line"),
        Line2D([0], [0], color="darkred", linewidth=2.2, linestyle="--",
               alpha=0.5, label=f"Crisis threshold $\\tau = {tau:.3f}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return _save(fig, name)


# =========================================================
# fig10 — Regime-conditional regression
# =========================================================
def fig10_regime_conditional_scatter(scatter_df, fitted_models, tau=0.851,
                                     name="fig10_regime_conditional_scatter"):
    apply_style()
    regime_colors = REGIME_COLORS
    panel_info = {
        "Tracking_Error": ("Average tracking error",
                           r"$|r^{(f)}_{i,t} - r^{(b,f)}_{i,t}|$"),
        "MSCI_World":  ("MSCI World",  r"$r^{(b,g)}_{t}$ (MSCI World)"),
        "MSCI_Europe": ("MSCI Europe", r"$r^{(b,g)}_{t}$ (MSCI Europe)"),
        "SP500":       ("S&P 500",     r"$r^{(b,g)}_{t}$ (S&P 500)"),
    }
    panel_order = ["Tracking_Error", "MSCI_World", "MSCI_Europe", "SP500"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    ar_values = scatter_df["AR"].values

    for ax, col in zip(axes, panel_order):
        title, ylabel = panel_info[col]
        y_values = scatter_df[col].values
        res = fitted_models[col]

        for regime in ["Low-Risk", "Moderate-Risk", "High-Risk"]:
            mask = (scatter_df["Regime"] == regime).values
            ax.scatter(ar_values[mask], y_values[mask],
                       c=regime_colors[regime], s=14, alpha=0.55,
                       edgecolors="none", label=regime)

        b0, b1, b2, b3 = res.params
        calm_mask = (scatter_df["Regime"] != "High-Risk").values
        high_mask = (scatter_df["Regime"] == "High-Risk").values

        if calm_mask.any():
            grid_c = np.linspace(ar_values[calm_mask].min(),
                                 ar_values[calm_mask].max(), 100)
            ax.plot(grid_c, b0 + b1 * grid_c, color="#1F4E79", linewidth=2.2,
                    label=f"Calm fit (slope $b_1$={b1:.3f})", zorder=5)
        if high_mask.any():
            grid_h = np.linspace(ar_values[high_mask].min(),
                                 ar_values[high_mask].max(), 100)
            ax.plot(grid_h, (b0 + b2) + (b1 + b3) * grid_h,
                    color="#8B0000", linewidth=2.2,
                    label=f"High-Risk fit (slope $b_1+b_3$={b1+b3:.3f})",
                    zorder=5)

        ax.axvline(tau, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_xlabel("$AR_t$", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=28)

        handles, labels = ax.get_legend_handles_labels()
        line_h = [h for h, l in zip(handles, labels) if "fit" in l.lower()]
        line_l = [l for l in labels if "fit" in l.lower()]
        ax.legend(line_h, line_l, loc="upper center",
                  bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)

    fig.tight_layout()
    return _save(fig, name)


# =========================================================
# fig11 — Risk decomposition
# =========================================================
def fig11_risk_decomposition(risk_decomp, regime_series, ar_index,
                             name="fig11_risk_decomposition"):
    apply_style()
    aligned = risk_decomp.reindex(ar_index).dropna()
    regimes = regime_series.reindex(aligned.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"hspace": 0.15})

    ax1.fill_between(aligned.index, 0, aligned["AR_t"],
                     alpha=0.4, color="#d73027",
                     label="Systematic share ($AR_t$)")
    ax1.fill_between(aligned.index, aligned["AR_t"], 1,
                     alpha=0.4, color="#2166ac",
                     label="Specific share ($1 - AR_t$)")
    ax1.set_ylabel("Risk share", fontsize=11)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=9, loc="lower right")
    _shade_regimes([ax1], regimes, aligned.index,
                   alpha_main=0.08, alpha_panel=0.08)

    ax2.plot(aligned.index, aligned["Total_Risk"]      * 1e4,
             color="black",   linewidth=1.0, label="Total risk")
    ax2.plot(aligned.index, aligned["Systematic_Risk"] * 1e4,
             color="#d73027", linewidth=0.8, alpha=0.8,
             label="Systematic risk")
    ax2.plot(aligned.index, aligned["Specific_Risk"]   * 1e4,
             color="#2166ac", linewidth=0.8, alpha=0.8,
             label="Specific risk")
    ax2.set_ylabel("Risk ($\\times 10^{-4}$)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    _shade_regimes([ax2], regimes, aligned.index,
                   alpha_main=0.08, alpha_panel=0.08)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, name)


# =========================================================
# fig12 — Covariance estimator comparison
# =========================================================
def fig12_covariance_comparison(comparison_df,
                                name="fig12_covariance_comparison"):
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    style_map = {
        "Sample":      ("black",   0.9, "Sample covariance (baseline)"),
        "Ledoit_Wolf": ("#2166ac", 0.7, "Ledoit-Wolf shrinkage"),
        "MCD":         ("#d73027", 0.7, "Minimum Covariance Determinant"),
    }
    for col, (color, lw, label) in style_map.items():
        if col in comparison_df.columns:
            ax.plot(comparison_df.index, comparison_df[col],
                    color=color, linewidth=lw, alpha=0.85, label=label)

    ax.set_ylabel("$AR_t$", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return _save(fig, name)
