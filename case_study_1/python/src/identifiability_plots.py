import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor.tensor as pt

# Set up subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()

# Loop over sigma_delta values
for i, sd in enumerate([0.01, 0.05, 0.1, 0.2, 0.3]):
    print(f"Processing σ of δ = {sd}")
    axs[i].set_title(f"σ of δ = {sd}", fontsize=20)
    axs[i].set_xlabel("μ and δ (/day)", fontsize=20)
    axs[i].set_ylabel("PDF", fontsize=20)
    axs[i].tick_params(axis='both', labelsize=18)

    # Load trace
    file_path = f"./../data/trace_sigma_delta_{sd}.nc"
    trace = az.from_netcdf(file_path)

    # Extract posterior samples
    mum_posterior = trace.posterior['mum'].values.flatten()

    # Smooth KDE for posterior (red line)
    kde_x, kde_y = az.kde(mum_posterior)
    posterior_line, = axs[i].plot(
        kde_x,
        kde_y,
        color='red',
        linewidth=2.5,
        label='Posterior of μ'
    )

    # Evaluate prior distribution
    x = np.linspace(0.01, 1.0, 300)
    delta_prior_dist = pm.TruncatedNormal.dist(
        mu=0.2, sigma=sd, lower=0.01, upper=1.0
    )
    prior_pdf = np.exp(pm.logp(delta_prior_dist, x).eval())

    prior_line, = axs[i].plot(
        x,
        prior_pdf,
        color='black',
        linewidth=2,
        label='Prior of δ'
    )

# Use the last (empty) subplot for legend only
axs[-1].axis('off')
axs[-1].legend(
    handles=[posterior_line, prior_line],
    loc='center',
    fontsize=20,
    frameon=False
)

fig.tight_layout()
plt.show()