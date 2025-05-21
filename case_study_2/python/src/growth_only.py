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


# ---------------------------
# Load the data and visualize
# ---------------------------



death_whole = pd.read_csv("./../data/ehux379_sytox.csv")
data = death_whole.head(10)





# Initial conditions with correct indexing
y0_guess = [
    (data['rep1'].iloc[0] + data['rep2'].iloc[0] + data['rep3'].iloc[0]) / 3
]


## differential equation and solvers

def cells_growth(y,t,r,K):
    P = y
    
    dydt = r*(1-P/K)*P 
   
    return dydt

def solved_num_cells(y0,t,r,K):
    sol = odeint(cells_growth, y0, t, args=(r,K))
    return sol

r_guess = 0.5
K_guess = 5e6


avg_data = data[['rep1', 'rep2', 'rep3']].mean(axis=1)
print(np.log(avg_data))

#initial conditions
t = np.linspace(0, 20, 100)

# guess solution
solution = solved_num_cells(y0_guess,t, r_guess, K_guess)



plt.plot(data['time (hours)'],data['rep1'],'o',color ='orange')
plt.plot(data['time (hours)'],data['rep2'],'o',color ='blue')
plt.plot(data['time (hours)'],data['rep3'],'o',color ='green')
plt.plot(t,solution[:,0],'-',color ='k')
plt.xlabel('Time (hrs)')
plt.ylabel(data.columns[1])
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)
plt.show()






# ---------------------------
# INFERENCE
# ---------------------------

@njit
def cells_growth(y,t,theta):
    P= y
    # unpack parameters
    r,K= theta 
   
    dydt = r*(1-P/K)*P 
   
    return dydt



@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix_vary_init(theta):
    y0 = [ theta[-1]]
    # Simulate ODE for each time series separately
    result = odeint(cells_growth, y0, t=data['time (hours)'].values, args=(theta[:-1],), rtol=1e-6, atol=1e-6)
   
    return result



import pytensor.tensor as pt


with pm.Model() as model:
    # Priors
    r = pm.Uniform(r"$r$ (growth rate)", lower=0.2, upper=0.7)
    K = pm.Uniform(r"$K$ (carrying capacity)" , lower=1e6, upper=2e7)
    
    P0 = pm.Uniform(r"$P_0$ (init. live)", lower=4.5e5, upper=9e5)
    

    sigma_ll = pm.HalfNormal(r"$\sigma_L$", 3)
    

    theta = pm.math.stack([r, K, P0])
    live_solution = pytensor_forward_model_matrix_vary_init(theta)

    # Likelihoods
    avg_data = data[['rep1', 'rep2', 'rep3']].mean(axis=1)

    #### I need to clip the values to avoid log(0) ####

    #pm.Normal("Y_live", mu=pm.math.log(live_solution[:, 0]), sigma=sigma_live,
    #          observed=np.log(avg_data.values))
    pm.Normal("Y_live", mu=pm.math.log(pm.math.clip(live_solution[:, 0], 1e-8, np.inf)),sigma=sigma_ll,
              observed=np.log(avg_data.values))
    
   
    
    
    
#model.debug()

# Specify the sampler
sampler = "Slice Sampler"
tune  = 10000
draws = 5000

# Variable list to give to the sample step parameter
vars_list = list(model.values_to_rvs.keys())[:-1]
print(vars_list)

with model:
    trace = pm.sample(tune=tune, draws=draws,target_accept=0.95, chains=4, cores=4)


# Plot chains
az.plot_trace(trace)
plt.show()

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
    posterior_samples["$r$ (growth rate)"].values,
    posterior_samples["$K$ (carrying capacity)"].values,
    posterior_samples["$P_0$ (init. live)"].values,
]).T  # Shape: (n_samples, 5)

n_plot = 200  # Number of samples to simulate for plotting

plt.figure(figsize=(12, 6))

for i in range(n_plot):
    theta = posterior_array[i]
    # last two are initial conditions
    y0 = [ theta[-1]]
    sol = odeint(cells_growth, y0, t=data['time (hours)'].values, args=(theta[:-1],), rtol=1e-6, atol=1e-6)


    plt.plot(data['time (hours)'], sol[:, 0], '-', color='gray', alpha=0.1)



# Add data points on top

plt.plot(data['time (hours)'], data['rep1'], 'o', color='orange')
plt.plot(data['time (hours)'], data['rep2'], 'o', color='blue')
plt.plot(data['time (hours)'], data['rep3'], 'o', color='green')
plt.xlabel('Time (hrs)')
plt.ylabel('Live cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)



plt.show()

# Save chain to a CSV file
df_trace = az.convert_to_inference_data(obj=trace).to_dataframe(include_coords=False,groups='posterior')
df_trace.to_csv('./../res/growth_only.csv', index=False)
