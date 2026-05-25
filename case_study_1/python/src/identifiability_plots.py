import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Colors (same as Part 1)
# ------------------------------------------------------------
hist_color = "C3"       # reddish histogram
mu_color = "#b22222"    # darker red

# ------------------------------------------------------------
# Typography
# ------------------------------------------------------------
increase = 4

plt.rcParams.update({
    "font.size": 16 + increase,
    "axes.titlesize": 18 + increase,
    "axes.labelsize": 17 + increase,
    "xtick.labelsize": 14 + increase,
    "ytick.labelsize": 14 + increase,
    "legend.fontsize": 14 + increase,
})

# ============================================================
# SETTINGS
# ============================================================

sigma_values = [0.01, 0.05, 0.1, 0.2, 0.3]

posterior_mu_stds = []

# ============================================================
# FIGURE
# ============================================================

fig, axs = plt.subplots(
    2,
    3,
    figsize=(18, 11),
    constrained_layout=True
)

axs = axs.flatten()

# ============================================================
# FIRST 5 PANELS: HISTOGRAMS
# ============================================================

for i, sd in enumerate(sigma_values):

    print(f"Processing σ of δ = {sd}")

    ax = axs[i]

    # --------------------------------------------------------
    # Load trace
    # --------------------------------------------------------
    file_path = f"./../data/trace_sigma_delta_{sd}.nc"
    trace = az.from_netcdf(file_path)

    # --------------------------------------------------------
    # Posterior samples
    # --------------------------------------------------------
    mum_posterior = (
        trace.posterior["mum"]
        .values
        .flatten()
    )

    posterior_mean = np.mean(mum_posterior)
    posterior_std = np.std(mum_posterior)

    posterior_mu_stds.append(
        posterior_std
    )

    # --------------------------------------------------------
    # Posterior histogram of μ
    # --------------------------------------------------------
    ax.hist(
        mum_posterior,
        bins=40,
        range=(0, 1),
        density=True,
        color=hist_color,
        alpha=0.28,
        edgecolor="none"
    )

    # --------------------------------------------------------
    # Prior of δ
    # --------------------------------------------------------
    x = np.linspace(
        0.01,
        1.0,
        400
    )

    delta_prior_dist = pm.TruncatedNormal.dist(
        mu=0.2,
        sigma=sd,
        lower=0.01,
        upper=1.0
    )

    prior_pdf = np.exp(
        pm.logp(
            delta_prior_dist,
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
        upper=1.0
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
        posterior_mean,
        color=mu_color,
        linewidth=3,
        linestyle="--",
        alpha=0.95
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------
    ax.set_title(
        rf"Prior $\sigma$ of $\delta$ = {sd}",
        pad=10
    )

    ax.set_xlabel(
        r"$\mu$ and $\delta$ (/day)"
    )

    ax.set_ylabel(
        "PDF"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_box_aspect(1)

# ============================================================
# SIXTH PANEL:
# Posterior std dev of μ
# ============================================================

ax_std = axs[-1]

ax_std.plot(
    sigma_values,
    posterior_mu_stds,
    "o",
    color=mu_color,
    markersize=9,
    zorder=3
)



# ------------------------------------------------------------
# Labels and formatting
# ------------------------------------------------------------

ax_std.set_xlabel(
    r"Prior Std Dev of $\delta$ (/day)"
)

ax_std.set_ylabel(
    r"Posterior Std Dev of $\mu$ (/day)"
)

ax_std.set_xlim(
    -0.005,
    0.32
)

ax_std.set_box_aspect(1)

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
    bbox_to_anchor=(0.5, 0.02),
    ncol=4,
    frameon=False
)

# ============================================================
# SAVE / SHOW
# ============================================================

plt.savefig(
     "./../../../figures/identifiability_sigma_sweep.svg",
     bbox_inches="tight")

plt.show()