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
    #axs[i].grid(True)

    # Load trace
    file_path = f"./../data/trace_sigma_delta_{sd}.nc"
    trace = az.from_netcdf(file_path)

    # Extract posterior samples
    mum_posterior = trace.posterior['mum'].values.flatten()
    axs[i].hist(mum_posterior, bins=50, density=True, alpha=0.5, color='red', label='Posterior of μ')

    # Evaluate prior distribution
    x = np.linspace(0.01, 1.0, 100)
    delta_prior_dist = pm.TruncatedNormal.dist(mu=0.2, sigma=sd, lower=0.01, upper=1.0)
    prior_pdf = np.exp(pm.logp(delta_prior_dist, x).eval())

    axs[i].plot(x, prior_pdf, color='black', label='Prior of δ', linewidth=2)
    axs[i].legend(fontsize=18)

# Hide unused subplot
if len(axs) > len([0.01, 0.05, 0.1, 0.2, 0.3]):
    axs[-1].axis('off')

fig.tight_layout()
plt.show()