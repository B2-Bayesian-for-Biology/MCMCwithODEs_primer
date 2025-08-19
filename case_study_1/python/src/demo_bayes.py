import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pymc as pm
import arviz as az
from scipy.stats import norm

# -------------------------------
# ODE definition
# -------------------------------
def cells_ode(y, t, params):
    mum = params[0]
    return [mum * y[0]]

def solve_ode(mum, N0, time):
    sol = odeint(cells_ode, [N0], time, args=([mum],))
    return sol[:, 0]

# -------------------------------
# Load data
# -------------------------------
data = pd.read_csv("./../data/phaeocystis_control.csv")
time = data['times'].values
obs = data['cells'].values
N0_fixed = obs[0]

# -------------------------------
# Manual Bayes calculation
# -------------------------------
# Define prior on mum
prior_mu = 0.5
prior_sigma = 0.3
prior = lambda x: norm.pdf(x, loc=prior_mu, scale=prior_sigma)

# Likelihood: assume Gaussian noise on log-scale
sigma_obs = 0.2  # assumed observational noise
def likelihood(mum):
    pred = solve_ode(mum, N0_fixed, time)
    return np.prod(norm.pdf(np.log(obs), loc=np.log(pred), scale=sigma_obs))

# Posterior grid
mum_grid = np.linspace(0, 1.5, 200)
prior_vals = prior(mum_grid)
likelihood_vals = np.array([likelihood(m) for m in mum_grid])

unnormalized_posterior = prior_vals * likelihood_vals
posterior_vals = unnormalized_posterior / np.trapz(unnormalized_posterior, mum_grid)

# -------------------------------
# PyMC MCMC inference
# -------------------------------
with pm.Model() as model:
    mum = pm.Normal('mum', mu=prior_mu, sigma=prior_sigma)
    sigma = pm.HalfNormal("sigma", 1.0)

    def ode_system(y, t, theta):
        return [theta[0] * y[0]]

    cell_model = pm.ode.DifferentialEquation(
        func=ode_system,
        times=time,
        n_states=1,
        n_theta=1,
        t0=0
    )

    y_hat = cell_model(y0=[N0_fixed], theta=[mum])
    pm.Normal("Y_obs", mu=pm.math.log(y_hat[:, 0]), sigma=sigma_obs, observed=np.log(obs))

    trace = pm.sample(draws=2000, tune=1000, chains=2, return_inferencedata=True, target_accept=0.9)

# -------------------------------
# Plot prior, likelihood, posterior vs MCMC
# -------------------------------
plt.figure(figsize=(8,6))

# Scale likelihood for plotting (since absolute likelihood can be tiny)
likelihood_scaled = likelihood_vals / np.max(likelihood_vals) * np.max(prior_vals)

plt.plot(mum_grid, prior_vals, label="Prior", lw=2, color="black")
plt.plot(mum_grid, likelihood_scaled, label="Likelihood (scaled)", lw=2)
plt.plot(mum_grid, posterior_vals, label="Posterior (analytical)", lw=2)

# Plot MCMC posterior
samples = trace.posterior["mum"].values.flatten()
plt.hist(samples, bins=50, density=True, alpha=0.4, color="C3", label="Posterior (MCMC)")

plt.xlabel(r"$\mu$")
plt.ylabel("Probability Density")
plt.legend()
plt.title(r"Bayesian inference demo: Posterior $\propto$ Prior $\times$ Likelihood")
plt.tight_layout()
plt.show()