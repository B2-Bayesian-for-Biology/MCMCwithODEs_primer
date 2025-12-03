from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv("./../data/phaeocystis_control.csv")
t_data = data['times'].values
y_data = data['cells'].values
y0 = y_data[0]  # Initial condition, assuming the first value is the initial cell count


def exp_model(t, mum):
    return y0 * np.exp(mum * t)

popt, pcov = curve_fit(exp_model, t_data, y_data)
mum_lsq = popt[0]
print(f"Least Squares Estimate of mum: {mum_lsq}")

from scipy.optimize import minimize

def neg_log_likelihood(mum):
    y_model = y0 * np.exp(mum * t_data)
    sigma = 0.1  # assume known or fit
    return -np.sum(-0.5 * ((y_data - y_model) / sigma) ** 2 - np.log(sigma) - 0.5*np.log(2*np.pi))

res = minimize(neg_log_likelihood, x0=[0.3])
mum_mle = res.x[0]
print(f"Maximum Likelihood Estimate of mum: {mum_mle}")

def neg_log_posterior(mum):
    # Prior: Normal(0.2, 0.1)
    mum_prior = 0.5
    sigma_prior = 0.3
    log_prior = -0.5 * ((mum - mum_prior) / sigma_prior)**2 - np.log(sigma_prior) - 0.5*np.log(2*np.pi)
    return -log_prior + neg_log_likelihood(mum)

res = minimize(neg_log_posterior, x0=[0.2])
mum_map = res.x[0]
print(f"MAP Estimate of mum: {mum_map}")

from sklearn.utils import resample
bootstrap_estimates = []
for _ in range(1000):
    t_bs, y_bs = resample(t_data, y_data)
    popt, _ = curve_fit(exp_model, t_bs, y_bs)
    bootstrap_estimates.append(popt[0])

print(f"Bootstrap Estimate of mum: {np.mean(bootstrap_estimates)}" and f"Standard Deviation: {np.std(bootstrap_estimates)}")


mus = np.linspace(0.01, 1.0, 500)
log_likelihoods = [-neg_log_likelihood(mu) for mu in mus]

'''
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(mus, log_likelihoods, label='Log-Likelihood')
plt.axvline(mum_lsq, color='r', linestyle='--', label='Least Squares Estimate')
plt.axvline(mum_mle, color='g', linestyle='--', label='MLE Estimate')
plt.axvline(mum_map, color='b', linestyle='--', label='MAP Estimate')
plt.axvline(np.mean(bootstrap_estimates), color='orange', linestyle='--', label='Bootstrap Estimate')
plt.title('Log-Likelihood vs. Growth Rate Estimates')
plt.xlabel('Growth Rate (mum)')
plt.ylabel('Log-Likelihood')
plt.legend()
plt.grid()
plt.show()
plt.savefig('./../figures/log_likelihood_vs_growth_rate.png')

import matplotlib.pyplot as plt
import numpy as np

import arviz as az
# Load the MCMC trace
trace = az.from_netcdf('./../data/normal_growth_trace.nc')


# Estimates to plot
estimates = {
    'Least Squares': mum_lsq,
    'MLE': mum_mle,
    'MAP': mum_map,
    'Bootstrap': np.mean(bootstrap_estimates),
    'Posterior Mean': np.mean(trace.posterior['mum'].values)
}

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(estimates.keys(), estimates.values(), color=['red', 'green', 'blue', 'orange', 'purple'])
plt.ylabel('Estimated Growth Rate (mum)')
plt.title('Comparison of Growth Rate Estimates from Different Methods')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Save and show
plt.tight_layout()
plt.savefig('./../figures/growth_rate_estimates_comparison.png')
plt.show()

'''

import matplotlib.pyplot as plt
import numpy as np
import arviz as az

# Load the MCMC trace
trace = az.from_netcdf('./../data/normal_growth_trace.nc')

# Extract posterior samples
posterior_samples = trace.posterior['mum'].values.flatten()
posterior_mean = np.mean(posterior_samples)
posterior_std = np.std(posterior_samples)

# Bootstrap statistics
bootstrap_mean = np.mean(bootstrap_estimates)
bootstrap_std = np.std(bootstrap_estimates)

# Method names and values
methods = ['Least Squares', 'MLE', 'MAP', 'Bootstrap', 'Posterior Mean']
estimates = [mum_lsq, mum_mle, 0.2513 , bootstrap_mean,0.2509]  #posterior_mean mum_map
errors = [0, 0, 0, bootstrap_std, 0.0390]  # Only last two have error bars posterior_std

# Plot
plt.figure(figsize=(10, 6))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.errorbar(methods, estimates, yerr=errors, fmt='o', capsize=5, color='black')
plt.ylabel('Estimated Growth Rate μ (/day)', fontsize=20)
#plt.title('Comparison of Growth Rate Estimates from Different Methods')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('./../figures/growth_rate_estimates_dotplot.png')
plt.show()