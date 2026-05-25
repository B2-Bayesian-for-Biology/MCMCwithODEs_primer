

# ============================================================
# FIGURE LAYOUT (2 x 4)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

hist_color = "C3"       # reddish histogram
mu_color = "#b22222"    # slightly darker/redder than C3

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "./../data"

# Must match values used during inference
prior_mean_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# Fixed prior width used during inference
SIGMA_DELTA = 0.1

increase = 6

# ------------------------------------------------------------
# Typography
# ------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16 + increase,
    "axes.titlesize": 18 + increase,
    "axes.labelsize": 17 + increase,
    "xtick.labelsize": 14 + increase,
    "ytick.labelsize": 14 + increase,
    "legend.fontsize": 14 + increase,
})

# ============================================================
# LOAD TRACES
# ============================================================

traces = {}

posterior_mu_means = []
posterior_mu_stds = []

posterior_delta_means = []
posterior_delta_stds = []

mu_minus_delta_means = []

print("\nLoading traces...\n")

for pmd in prior_mean_values:

    fname = os.path.join(
        DATA_DIR,
        f"trace_mean_delta_{pmd}.nc"
    )

    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"Missing file:\n{fname}"
        )

    trace = az.from_netcdf(fname)
    traces[pmd] = trace

    # --------------------------------------------------------
    # Posterior of mu
    # --------------------------------------------------------
    mum_post = trace.posterior["mum"].values.flatten()

    mu_mean = float(np.mean(mum_post))
    mu_std = float(np.std(mum_post))

    posterior_mu_means.append(mu_mean)
    posterior_mu_stds.append(mu_std)

    # --------------------------------------------------------
    # Posterior of delta
    # Change variable name if needed
    # --------------------------------------------------------
    delta_post = trace.posterior["delta"].values.flatten()

    delta_mean = float(np.mean(delta_post))
    delta_std = float(np.std(delta_post))

    posterior_delta_means.append(delta_mean)
    posterior_delta_stds.append(delta_std)

    # --------------------------------------------------------
    # Mean(mu - delta)
    # --------------------------------------------------------
    mu_minus_delta = np.mean(
        mum_post - delta_post
    )

    mu_minus_delta_means.append(
        float(mu_minus_delta)
    )

    print(
        f"δ prior mean = {pmd:.2f} | "
        f"μ mean = {mu_mean:.4f} | "
        f"δ mean = {delta_mean:.4f} | "
        f"μ−δ = {mu_minus_delta:.4f}"
    )

# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(22, 12),
    constrained_layout=True
)

axes = axes.flatten()

# ============================================================
# TOP 7 PANELS: HISTOGRAMS
# ============================================================

for idx, pmd in enumerate(prior_mean_values):

    ax = axes[idx]

    trace = traces[pmd]

    mum_post = (
        trace.posterior["mum"]
        .values
        .flatten()
    )

    # --------------------------------------------------------
    # Posterior histogram of μ
    # --------------------------------------------------------
    ax.hist(
        mum_post,
        bins=40,
        range=(0, 1.5),
        density=True,
        #color="lightgrey",
        #alpha=0.9,
        color=hist_color,
        alpha=0.28,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Prior of δ
    # --------------------------------------------------------
    x = np.linspace(
        0.01,
        1.5,
        400
    )

    delta_dist = pm.TruncatedNormal.dist(
        mu=pmd,
        sigma=SIGMA_DELTA,
        lower=0.01,
        upper=1.5
    )

    prior_pdf = np.exp(
        pm.logp(
            delta_dist,
            x
        ).eval()
    )

    ax.plot(
        x,
        prior_pdf,
        color="black",
        linewidth=3
    )


    # --------------------------------------------------------
    # Prior of μ
    # --------------------------------------------------------

    mu_dist = pm.TruncatedNormal.dist(
        mu=0.5,
        sigma=0.3,
        lower=0.01,
        upper=1.5
    )

    mu_prior_pdf = np.exp(
        pm.logp(
            mu_dist,
            x
        ).eval()
    )

    ax.plot(
        x,
        mu_prior_pdf,
        color=mu_color,
        linewidth=3,
        alpha=0.9
    )


    # --------------------------------------------------------
    # Posterior mean line
    # --------------------------------------------------------
    ax.axvline(
        posterior_mu_means[idx],
        #color="dimgray",
        color=mu_color,
        linewidth=3,
        linestyle="--",
        alpha=0.95
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------
    ax.set_title(
        rf"Prior mean of $\delta$ = {pmd}",
        pad=10
    )

    ax.set_xlabel(
        r"$\mu$ and $\delta$ (/day)"
    )

    if idx % 2 == 0:
        ax.set_ylabel("PDF")

    ax.set_xlim(0, 1.5)

    # Square panel
    ax.set_box_aspect(1)

# ============================================================
# GLOBAL LEGEND
# ============================================================
legend_handles = [

    Line2D(
        [0], [0],
        color="black",
        linewidth=3,
        label=r"Prior of $\delta$"
    ),

    Line2D(
        [0], [0],
        color=mu_color,
        linewidth=3,
        label=r"Prior of $\mu$"
    ),

    Patch(
        facecolor=hist_color,
        alpha=0.28,
        label=r"Posterior of $\mu$"
    ),

    Line2D(
        [0], [0],
        color=mu_color,
        linewidth=3,
        linestyle="--",
        label=r"Posterior mean of $\mu$"
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.4, 0.1),
    ncol=4,
    frameon=False
)

# ============================================================
# LAST PANEL
# ============================================================

ax_mean = axes[-1]

# ------------------------------------------------------------
# μ posterior mean
# ------------------------------------------------------------
ax_mean.plot(
    prior_mean_values,
    posterior_mu_means,
    "o",
    color=mu_color,
    markersize=9,
    label=r"Mean of Posterior of $\mu$",
    zorder=3
)

# ------------------------------------------------------------
# μ − δ posterior mean
# ------------------------------------------------------------
ax_mean.plot(
    prior_mean_values,
    mu_minus_delta_means,
    "s",
    color="grey",
    markersize=8,
    label=r"Mean of Posterior of $\mu-\delta$",
    zorder=3
)

# ------------------------------------------------------------
# SLOPE-1 REFERENCE LINE
# ------------------------------------------------------------

xline = np.array([
    min(prior_mean_values),
    max(prior_mean_values)
])

# approximate constant offset
net_rate_approx = (
    posterior_mu_means[0]
    - prior_mean_values[0]
)

ax_mean.plot(
    xline,
    xline + net_rate_approx,
    "k--",
    linewidth=2.5,
    alpha=0.9,
    label="y=x+c reference"
)

# ------------------------------------------------------------
# Constant reference for μ − δ
# ------------------------------------------------------------

constant_ref = mu_minus_delta_means[0]

ax_mean.axhline(
    constant_ref,
    color="grey",
    linestyle=":",
    linewidth=2.5,
    alpha=0.9,
    label=r"Constant $\mu-\delta$"
)

# ------------------------------------------------------------
# Square axes with proper slope perception
# ------------------------------------------------------------

xmin = min(prior_mean_values) - 0.02
xmax = max(prior_mean_values) + 0.02

ymin = min(
    min(posterior_mu_means),
    min(mu_minus_delta_means)
) - 0.02

ymax = max(
    max(posterior_mu_means),
    max(mu_minus_delta_means)
) + 0.02

lim_min = min(
    xmin,
    ymin - net_rate_approx
)

lim_max = max(
    xmax,
    ymax - net_rate_approx
)

ax_mean.set_xlim(
    lim_min,
    lim_max
)

#ax_mean.set_ylim(
#    lim_min + net_rate_approx,
#    lim_max + net_rate_approx
#)

ax_mean.set_ylim(
    0.1,
    0.9
)

ax_mean.set_xlim(
    -0.02, 0.6)
# ------------------------------------------------------------
# Add y-tick for constant μ−δ value
# ------------------------------------------------------------


current_yticks = np.array(
    ax_mean.get_yticks()
)

# remove auto tick too close to constant_ref
threshold = 0.03

filtered_ticks = [
    tick for tick in current_yticks
    if abs(tick - constant_ref) > threshold
]

# add constant reference tick
yticks_new = np.sort(
    np.array(
        filtered_ticks + [constant_ref]
    )
)

ax_mean.set_yticks(
    yticks_new
)

# custom labels
yticklabels = []

for tick in yticks_new:

    if np.isclose(
        tick,
        constant_ref,
        atol=1e-3
    ):
        yticklabels.append(
            rf"{constant_ref:.2f}"
        )
    else:
        yticklabels.append(
            f"{tick:.1f}"
        )

ax_mean.set_yticklabels(
    yticklabels
)

ax_mean.set_aspect(
    "equal",
    adjustable="box"
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax_mean.set_xlabel(
    r"Prior mean of $\delta$ (/day)"
)

ax_mean.set_ylabel(
    r"Posterior mean (/day)"
)


#ax_mean.legend(
#    loc="best",
#    frameon=False
#)

ax_mean.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    frameon=False,
    ncol=1
)


# Make square
ax_mean.set_box_aspect(1)


# SAVE
# ============================================================

save_svg = os.path.join(
    DATA_DIR,
    "identifiability_prior_mean_sweep_replot.svg"
)

save_png = os.path.join(
    DATA_DIR,
    "identifiability_prior_mean_sweep_replot.png"
)

plt.savefig(
    save_svg,
    bbox_inches="tight"
)

plt.savefig(
    save_png,
    dpi=300,
    bbox_inches="tight"
)



print("\nFigure saved:")
print(save_svg)
print(save_png)