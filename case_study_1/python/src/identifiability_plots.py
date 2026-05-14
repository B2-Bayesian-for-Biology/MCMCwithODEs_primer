import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm

# Set up subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

# Loop over sigma_delta values
for i, sd in enumerate([0.01, 0.05, 0.1, 0.2, 0.3]):
    print(f"Processing σ of δ = {sd}")

    ax = axs[i]
    ax.set_title(f"σ of δ = {sd}", fontsize=20)
    ax.set_xlabel("μ and δ (/day)", fontsize=20)
    ax.set_ylabel("PDF", fontsize=20)
    ax.tick_params(axis='both', labelsize=18)

    # Load trace
    file_path = f"./../data/trace_sigma_delta_{sd}.nc"
    trace = az.from_netcdf(file_path)

    # Extract posterior samples
    mum_posterior = trace.posterior["mum"].values.flatten()

    # ---- Posterior: normalized histogram ----
    posterior_hist = ax.hist(
        mum_posterior,
        bins=40,
        range=(0, 1),
        density=True,
        color="red",
        alpha=0.6,
        edgecolor="none",
        label="Posterior of μ"
    )

    ax.set_xlim(0, 1)

    # ---- Prior: analytical PDF ----
    x = np.linspace(0.01, 1.0, 400)
    delta_prior_dist = pm.TruncatedNormal.dist(
        mu=0.2, sigma=sd, lower=0.01, upper=1.0
    )
    prior_pdf = np.exp(pm.logp(delta_prior_dist, x).eval())

    prior_line, = ax.plot(
        x,
        prior_pdf,
        color="black",
        linewidth=2.5,
        label="Prior of δ"
    )

# Use last subplot for legend only
axs[-1].axis("off")
axs[-1].legend(
    handles=[
        plt.Rectangle((0, 0), 1, 1, color="red", alpha=0.6),
        prior_line,
    ],
    labels=["Posterior of μ", "Prior of δ"],
    loc="center",
    fontsize=20,
    frameon=False,
)

fig.tight_layout()
#plt.savefig("./../../../figures/identifiability_plots_histogram.svg")
plt.show()

