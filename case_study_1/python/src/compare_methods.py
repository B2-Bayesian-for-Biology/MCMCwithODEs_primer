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
