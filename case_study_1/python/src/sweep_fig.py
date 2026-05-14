
# ============================================================
# FIGURE LAYOUT (4 x 2)
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import arviz as az
import pymc as pm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "./../data"

# Must match the values used during inference
prior_mean_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# Fixed prior width used during inference
SIGMA_DELTA = 0.1

# ============================================================
# LOAD TRACES
# ============================================================

traces = {}
posterior_means = []
posterior_stds = []

print("\nLoading traces...\n")

for pmd in prior_mean_values:

    fname = os.path.join(
        DATA_DIR,
        f"trace_mean_delta_{pmd}.nc"
    )

    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing file:\n{fname}")

    trace = az.from_netcdf(fname)
    traces[pmd] = trace

    mum_post = trace.posterior["mum"].values.flatten()

    mu_mean = float(np.mean(mum_post))
    mu_std = float(np.std(mum_post))

    posterior_means.append(mu_mean)
    posterior_stds.append(mu_std)

    print(
        f"δ prior mean = {pmd:.2f} | "
        f"μ posterior mean = {mu_mean:.4f} | "
        f"std = {mu_std:.4f}"
    )

increase = 6
# ---- typography -----------------------------------------------------------
plt.rcParams.update({
    "font.size": 16+increase,
    "axes.titlesize": 18+increase,
    "axes.labelsize": 17+increase,
    "xtick.labelsize": 14+increase,
    "ytick.labelsize": 14+increase,
    "legend.fontsize": 14+increase,
})

fig, axes = plt.subplots(
    2,
    4,
    figsize=(22, 12),
    constrained_layout=True
)

axes = axes.flatten()

# colour map
colours = plt.cm.coolwarm(
    np.linspace(0.1, 0.9, len(prior_mean_values))
)

# ============================================================
# TOP 7 PANELS: HISTOGRAMS
# ============================================================

for idx, (pmd, colour) in enumerate(
    zip(prior_mean_values, colours)
):

    ax = axes[idx]

    trace = traces[pmd]
    mum_post = trace.posterior["mum"].values.flatten()

    # --------------------------------------------------------
    # Posterior histogram of μ
    # --------------------------------------------------------
    ax.hist(
        mum_post,
        bins=40,
        range=(0, 1.5),
        density=True,
        color=colour,
        alpha=0.75,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Prior of δ
    # --------------------------------------------------------
    x = np.linspace(0.01, 1.5, 400)

    delta_dist = pm.TruncatedNormal.dist(
        mu=pmd,
        sigma=SIGMA_DELTA,
        lower=0.01,
        upper=1.5
    )

    prior_pdf = np.exp(
        pm.logp(delta_dist, x).eval()
    )

    ax.plot(
        x,
        prior_pdf,
        color="black",
        linewidth=3
    )

    # --------------------------------------------------------
    # Posterior mean line
    # --------------------------------------------------------
    ax.axvline(
        posterior_means[idx],
        color=colour,
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

    ax.set_xlabel(r"$\mu$ and $\delta$ (/day)")

    if idx % 2 == 0:
        ax.set_ylabel("PDF")

    ax.set_xlim(0, 1.5)
    # Make histogram panel square

    ax.set_box_aspect(1)

# ============================================================
# LEGEND (GLOBAL)
# ============================================================

legend_handles = [

    Line2D(
        [0], [0],
        color="black",
        linewidth=3,
        label=r"Prior of $\delta$"
    ),

    Patch(
        facecolor="grey",
        alpha=0.6,
        label=r"Posterior of $\mu$"
    ),

    Line2D(
        [0], [0],
        color="grey",
        linewidth=3,
        linestyle="--",
        label=r"Posterior mean of $\mu$"
    )
]

fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=True
)

# ============================================================
# LAST PANEL:
# BIAS IN μ
# ============================================================

ax_mean = axes[-1]

ax_mean.scatter(
    prior_mean_values,
    posterior_means,
    color=colours,
    s=120,
    edgecolor="k",
    linewidth=1.2,
    zorder=3
)

# ------------------------------------------------------------
# slope-1 line
# ------------------------------------------------------------

xline = np.array([
    min(prior_mean_values),
    max(prior_mean_values)
])

net_rate_approx = (
    posterior_means[0]
    - prior_mean_values[0]
)

ax_mean.plot(
    xline,
    xline + net_rate_approx,
    "k--",
    linewidth=2.5,
    label="Slope-1 reference"
)

# ------------------------------------------------------------
# square axes
# ------------------------------------------------------------

xmin = min(prior_mean_values) - 0.02
xmax = max(prior_mean_values) + 0.02

ymin = min(posterior_means) - 0.02
ymax = max(posterior_means) + 0.02

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

ax_mean.set_ylim(
    lim_min + net_rate_approx,
    lim_max + net_rate_approx
)

ax_mean.set_aspect(
    "equal",
    adjustable="box"
)

# ------------------------------------------------------------
# labels
# ------------------------------------------------------------

ax_mean.set_xlabel(
    r"Prior mean of $\delta$ (/day)"
)

ax_mean.set_ylabel(
    r"Posterior mean of $\mu$ (/day)"
)

ax_mean.set_title(
    r"Bias in $\mu$ from misspecified $\delta$ prior",
    pad=12
)

ax_mean.legend(
    loc="upper left",
    frameon=True
)

# ============================================================
# SUPTITLE
# ============================================================



# ============================================================
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

plt.show()

print("\nFigure saved:")
print(save_svg)
print(save_png)

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import arviz as az
import pymc as pm

# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "./../data"

# Must match the values used during inference
prior_mean_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# Fixed prior width used during inference
SIGMA_DELTA = 0.1

# ============================================================
# LOAD TRACES
# ============================================================

traces = {}
posterior_means = []
posterior_stds = []

print("\nLoading traces...\n")

for pmd in prior_mean_values:

    fname = os.path.join(
        DATA_DIR,
        f"trace_mean_delta_{pmd}.nc"
    )

    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing file:\n{fname}")

    trace = az.from_netcdf(fname)
    traces[pmd] = trace

    mum_post = trace.posterior["mum"].values.flatten()

    mu_mean = float(np.mean(mum_post))
    mu_std = float(np.std(mum_post))

    posterior_means.append(mu_mean)
    posterior_stds.append(mu_std)

    print(
        f"δ prior mean = {pmd:.2f} | "
        f"μ posterior mean = {mu_mean:.4f} | "
        f"std = {mu_std:.4f}"
    )

# ============================================================
# FIGURE LAYOUT
# ============================================================

n_panels = len(prior_mean_values)
ncols = n_panels

fig = plt.figure(figsize=(3.5 * ncols, 9))

gs = gridspec.GridSpec(
    2,
    ncols,
    figure=fig,
    hspace=0.45,
    wspace=0.35
)

# colour map
colours = plt.cm.coolwarm(
    np.linspace(0.1, 0.9, n_panels)
)

# ============================================================
# TOP ROW: HISTOGRAMS
# ============================================================

for col_idx, (pmd, colour) in enumerate(
    zip(prior_mean_values, colours)
):

    ax = fig.add_subplot(gs[0, col_idx])

    trace = traces[pmd]
    mum_post = trace.posterior["mum"].values.flatten()

    # Posterior of μ
    ax.hist(
        mum_post,
        bins=40,
        range=(0, 1.5),
        density=True,
        color=colour,
        alpha=0.75,
        edgecolor="none"
    )

    # Prior of δ
    x = np.linspace(0.01, 1.5, 400)

    delta_dist = pm.TruncatedNormal.dist(
        mu=pmd,
        sigma=SIGMA_DELTA,
        lower=0.01,
        upper=1.5
    )

    prior_pdf = np.exp(
        pm.logp(delta_dist, x).eval()
    )

    ax.plot(
        x,
        prior_pdf,
        color="black",
        linewidth=2,
        label="Prior of δ"
    )

    # Posterior mean of μ
    ax.axvline(
        posterior_means[col_idx],
        color=colour,
        linewidth=2,
        linestyle="--",
        alpha=0.9
    )

    ax.set_title(
        f"Prior mean of δ = {pmd}",
        fontsize=12
    )

    ax.set_xlabel(
        "μ and δ (/day)",
        fontsize=11
    )

    if col_idx == 0:
        ax.set_ylabel(
            "PDF",
            fontsize=11
        )

    ax.set_xlim(0, 1.5)
    ax.tick_params(labelsize=10)

# ============================================================
# LEGEND
# ============================================================

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_handles = [
    Line2D(
        [0], [0],
        color="black",
        linewidth=2,
        label="Prior of δ"
    ),
    Patch(
        facecolor="grey",
        alpha=0.6,
        label="Posterior of μ"
    ),
    Line2D(
        [0], [0],
        color="grey",
        linewidth=2,
        linestyle="--",
        label="Posterior mean of μ"
    )
    
    
]



fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.50),
    ncol=3,
    fontsize=11,
    frameon=True
)


# ============================================================
# BOTTOM (CENTERED):
# POSTERIOR MEAN OF μ
# ============================================================

ax_mean = fig.add_subplot(
    gs[1, 1:-1]
)

ax_mean.scatter(
    prior_mean_values,
    posterior_means,
    color=colours,
    s=80,
    zorder=3,
    edgecolor="k"
)

# ------------------------------------------------------------
# slope-1 reference line
# ------------------------------------------------------------

xline = np.array([
    min(prior_mean_values),
    max(prior_mean_values)
])

net_rate_approx = (
    posterior_means[0]
    - prior_mean_values[0]
)

ax_mean.plot(
    xline,
    xline + net_rate_approx,
    "k--",
    linewidth=1.5,
    label=(
       # f"Slope-1 ref\n"
       # f"(net rate ≈ {net_rate_approx:.2f})"
        f"Slope-1"
    )
)

# ------------------------------------------------------------
# FORCE SQUARE AXES + TRUE DIAGONAL
# ------------------------------------------------------------

# same limits for x and y so y=x+c is visually slope-1
xmin = min(prior_mean_values) - 0.02
xmax = max(prior_mean_values) + 0.02

ymin = min(posterior_means) - 0.02
ymax = max(posterior_means) + 0.02

# make common limits
lim_min = min(xmin, ymin - net_rate_approx)
lim_max = max(xmax, ymax - net_rate_approx)

ax_mean.set_xlim(lim_min, lim_max)
ax_mean.set_ylim(
    lim_min + net_rate_approx,
    lim_max + net_rate_approx
)

# make the plotting area square
ax_mean.set_aspect('equal', adjustable='box')

# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

ax_mean.set_xlabel(
    "Prior mean of δ (/day)",
    fontsize=13
)

ax_mean.set_ylabel(
    "Posterior mean of μ (/day)",
    fontsize=13
)

ax_mean.set_title(
    "Bias in μ from misspecified δ prior",
    fontsize=13
)

ax_mean.legend(fontsize=11)
ax_mean.tick_params(labelsize=11)



# ============================================================
# FINAL FORMATTING
# ============================================================

fig.suptitle(
    "Effect of prior mean of δ on inferred μ ",
    fontsize=14,
    y=1.01
)

# ============================================================
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
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved:")
print(save_svg)
print(save_png)
"""
