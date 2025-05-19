import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
import pymc as pm
from numba import njit
import pytensor.tensor as pt
import arviz as az
from pytensor.compile.ops import as_op
import logging
import os
import pytensor.tensor as pt

os.system('clear')

# ---------------------------
# Load the data and visualize
# ---------------------------



data = pd.read_csv("./../data/phaeocystis_control.csv")



# Initial conditions with correct indexing
y0_guess = [data['cells'].iloc[0]]


## differential equation and solvers
def cells_growth(y,t,r):
    P = y
    dydt = r*P 
    return dydt

def solved_num_cells(y0,t,r):
    sol = odeint(cells_growth, y0, t, args=(r,))
    return sol

r_guess = 0.34

#initial conditions
t = np.linspace(0, 2.625, 100)

# guess solution
solution = solved_num_cells(y0_guess,t, r_guess)


plt.figure(figsize=(12, 6))
plt.plot(data['times'],data['cells'],'o',color ='orange')
plt.plot(t,solution,'-',color ='k')
plt.xlabel('Time (hrs)')
plt.ylabel(data.columns[1])
plt.yscale('log')
plt.xlim(t[0], t[-1])
plt.show()





# ---------------------------
# INFERENCE
# ---------------------------

@njit
def cells_growth(y,t,r):
    P = y
    dydt = r*P 
    return dydt


@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix_vary_init(theta):
    y0 = [theta[-1]]
    # Simulate ODE for each time series separately
    result = odeint(cells_growth, y0, t=data['times'].values, args=(theta[0],), rtol=1e-6, atol=1e-6)
    return result




with pm.Model() as model:
    # Priors
    r = pm.Uniform(r"$r$ (effective growth rate)", lower=0.1, upper=0.5)
    P0 = pm.Uniform(r"$P_0$", lower=1e6, upper=3e6)

    sigma_ll = pm.HalfNormal(r"$\sigma_L$", 3)

    theta = pm.math.stack([r, P0])
    solution = pytensor_forward_model_matrix_vary_init(theta)
    pm.Normal("Y_live", mu=pm.math.log(solution[:,0]), sigma=sigma_ll, observed= np.log(data['cells'].values))
            

    
#model.debug()

# Specify the sampler
sampler = "Slice Sampler"
tune  = 2000
draws = 2000

# Variable list to give to the sample step parameter
vars_list = list(model.values_to_rvs.keys())[:-1]
print(vars_list)

with model:
    trace = pm.sample(tune=tune, draws=draws,target_accept=0.95, chains=4, cores=4)


import matplotlib.pyplot as plt
import arviz as az

# Plot trace
axes = az.plot_trace(trace)
chain_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# Loop through each axis and manually set line colors for each chain
for ax_row in axes:
    for ax in ax_row:
        lines = ax.get_lines()
        for i, line in enumerate(lines):
            line.set_color(chain_colors[i % len(chain_colors)])

plt.tight_layout()
plt.show()


# Plot posterior correlations
az.plot_pair(trace, kind='kde', divergences=True, marginals=True)
plt.show()


# 7 Convergence test for the chains - Gelman-Rubin, Geweke, and Autocorrelation
rhat = az.rhat(trace)
autocorr = az.plot_autocorr(trace)
print(f'Rhat:\n{rhat}\n')
plt.show()


# Get posterior samples as a NumPy array
posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
posterior_array = np.vstack([
    posterior_samples["$r$ (effective growth rate)"].values,
    posterior_samples["$P_0$"].values,
    
]).T  # Shape: (n_samples, 5)

n_plot = 200  # Number of samples to simulate for plotting

plt.figure(figsize=(12, 6))

for i in range(n_plot):
    theta = posterior_array[i]
    # last two are initial conditions
    y0 = [theta[-1]]
    solution = odeint(cells_growth, y0, t=data['times'].values, args=(theta[:-1],), rtol=1e-6, atol=1e-6)
    plt.plot(data['times'], solution[:, 0], '-', color='gray', alpha=0.1)

# Add data points on top

plt.plot(data['times'],data['cells'],'o',color ='blue')
plt.xlabel('Time (hrs)')
plt.ylabel(data.columns[1])
plt.yscale('log')
plt.title('Posterior predictive simulation')
plt.show()




# Save chain to a CSV file
df_trace = az.convert_to_inference_data(obj=trace).to_dataframe(include_coords=False,groups='posterior')
df_trace.to_csv('./../res/chain_results_v1.csv', index=False)
