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
# Define prior and likelihood
# -------------------------------
prior_mu = 0.5
prior_sigma = 0.1
prior_pdf = lambda x: norm.pdf(x, loc=prior_mu, scale=prior_sigma)

sigma_obs = 0.2  # assumed measurement noise (log-scale)

def likelihood_func(mum):
    pred = solve_ode(mum, N0_fixed, time)
    return np.prod(norm.pdf(np.log(obs), loc=np.log(pred), scale=sigma_obs))

# -------------------------------
# Compute grid values
# -------------------------------
mum_grid = np.linspace(0, 1.5, 400)

prior_vals = prior_pdf(mum_grid)
likelihood_vals_raw = np.array([likelihood_func(m) for m in mum_grid])

# Normalize likelihood so it becomes a PDF over mum
likelihood_vals = likelihood_vals_raw / np.trapz(likelihood_vals_raw, mum_grid)

# Posterior (unnormalized → normalized)
posterior_unnorm = prior_vals * likelihood_vals_raw
posterior_vals = posterior_unnorm / np.trapz(posterior_unnorm, mum_grid)

# -------------------------------
# PyMC MCMC inference
# -------------------------------
with pm.Model() as model:
    mum = pm.Normal('mum', mu=prior_mu, sigma=prior_sigma)

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
plt.rcParams.update({
    "font.size": 16,        # increase all font sizes
    "axes.labelsize": 18,   # x/y labels
    "axes.titlesize": 20,   # subplot titles
    "legend.fontsize": 14,  # legend text
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})
plt.figure(figsize=(8,6))

plt.plot(mum_grid, prior_vals, label="Prior", lw=2, color="black")
plt.plot(mum_grid, likelihood_vals, label="Likelihood", lw=2)
plt.plot(mum_grid, posterior_vals, label="Posterior (Analytical)", lw=2)

# Plot MCMC posterior as histogram
samples = trace.posterior["mum"].values.flatten()
plt.hist(samples, bins=50, density=True, alpha=0.4, color="C3", label="Posterior (MCMC)")

plt.xlabel(r"$\mu$")
plt.ylabel("Probability Density Function")
plt.legend()
plt.title("Prior, Likelihood, and Posterior")
plt.tight_layout()
plt.savefig("bayes_demo.png", dpi=300, bbox_inches="tight")
plt.show()