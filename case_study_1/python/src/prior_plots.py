import numpy as np
import arviz as az

import matplotlib.pyplot as plt

# Define the parameters for the Gaussian prior
mu_mean = 0.5
mu_std = 0.3

'''
# Generate samples from the normal distribution
mu_samples = np.random.normal(mu_mean, mu_std, 1000)

# Create an ArviZ InferenceData object
idata = az.from_dict(posterior={"mu": mu_samples})

# Plot the Gaussian prior
az.plot_posterior(idata, var_names=["mu"], hdi_prob=0.95)
plt.title("Gaussian Prior: mu ~ Normal(0.5, 0.3)")
plt.show()
'''


# Define the Gaussian function
def gaussian(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Generate x values for the plot
x_values = np.linspace(mu_mean - 3 * mu_std, mu_mean + 3 * mu_std, 100)

# Calculate the corresponding y values using the Gaussian function
y_values = gaussian(x_values, mu_mean, mu_std)

# Plot the Gaussian function
plt.plot(x_values, y_values, label='Gaussian Function', color='red')
plt.fill_between(x_values, y_values, alpha=0.2, color='red')
plt.legend()
plt.show()