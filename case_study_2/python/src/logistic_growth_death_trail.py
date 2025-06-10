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
death_data = death_whole.tail(10)



# Correct: use .iloc[0] to access by position
print(death_data['rep1'].iloc[0])

total_guess = (data['rep1'].iloc[0] + data['rep2'].iloc[0] + data['rep3'].iloc[0]) / 3
dead_guess = (death_data['rep1'].iloc[0] + death_data['rep2'].iloc[0] + death_data['rep3'].iloc[0]) / 3
# Initial conditions with correct indexing
y0_guess = [
total_guess - dead_guess, dead_guess
]

print(f"Initial conditions: {y0_guess}")


## differential equation and solvers

def cells_growth_death(y,t,r,K,delta):
    P, D = y
    dydt = [0,0]
    dydt[0] = r*(1-P/K)*P - delta*P
    dydt[1] = delta*P
    return dydt

def solved_num_cells(y0,t,r,K,delta):
    sol = odeint(cells_growth_death, y0, t, args=(r,K,delta))
    return sol

r_guess = 0.5
K_guess =3e6
delta_guess = 0.15


avg_data = data[['rep1', 'rep2', 'rep3']].mean(axis=1)
print(np.log(avg_data))

#initial conditions
t = np.linspace(0, 20, 100)

# guess solution
solution = solved_num_cells(y0_guess,t, r_guess, K_guess, delta_guess)


plt.subplot(1, 2, 1)
plt.plot(data['time (hours)'],data['rep1'],'o',color ='orange')
plt.plot(data['time (hours)'],data['rep2'],'o',color ='blue')
plt.plot(data['time (hours)'],data['rep3'],'o',color ='green')
plt.plot(t,solution[:,0]+solution[:,1],'-',color ='k')
plt.xlabel('Time (hrs)')
plt.ylabel(data.columns[1])
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)


plt.subplot(1, 2, 2)
plt.plot(death_data['time (hours)'],death_data['rep1'],'o',color ='orange')
plt.plot(death_data['time (hours)'],death_data['rep2'],'o',color ='blue')
plt.plot(death_data['time (hours)'],death_data['rep3'],'o',color ='green')
plt.plot(t,solution[:,1],'-',color ='k')
plt.xlabel(death_data.columns[0])
plt.ylabel('Cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)
plt.show()





# ---------------------------
# INFERENCE
# ---------------------------

@njit
def cells_growth_death(y,t,theta):
    P, D = y
    # unpack parameters
    r,K,delta = theta 
    dydt = [0,0]
    dydt[0] = r*(1-P/K)*P - delta*D
    dydt[1] = delta*D
    return dydt



@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix, pt.dmatrix])
def pytensor_forward_model_matrix_vary_init(theta):
    y0 = [theta[-2], theta[-1]]
    # Simulate ODE for each time series separately 
    # here time axis is same, but this would come handy if time axis were different
    result_1 = odeint(cells_growth_death, y0, t=data['time (hours)'].values, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
    result_2 = odeint(cells_growth_death, y0, t=death_data['time (hours)'].values, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
    # col 0 is of result_1 is live cells, col 1 of result_2 is dead cells
    return result_1, result_2



import pytensor.tensor as pt


with pm.Model() as model:
    # Priors
    r = pm.Uniform(r"$r$ (growth rate)", lower=0.25, upper=1)
    K = pm.Uniform(r"$K$ (carrying capacity)" , lower=2e6, upper=1e7)
    delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.05, upper=0.25)
    P0 = pm.Uniform(r"$P_0$ (init. live)", lower=3e5, upper=7e5)
    D0 = pm.Uniform(r"$D_0$ (init. dead)", lower=1e5, upper=4e5)

    sigma_live = pm.HalfNormal(r"$\sigma_L$", 3)
    sigma_dead = pm.HalfNormal(r"$\sigma_D$", 3)

    theta = pm.math.stack([r, K, delta, P0, D0])
    result_1, result_2 = pytensor_forward_model_matrix_vary_init(theta)

    live_solution = result_1[:,0]
    dead_solution = result_2[:,1]
    # Log likelihoods
    
    #### I need to clip the values to avoid log(0) ####
    avg_data = data[['rep1', 'rep2', 'rep3']].mean(axis=1)
    pm.Normal("Y_live", mu=pm.math.log(pm.math.clip(live_solution+dead_solution, 1e-8, np.inf)),sigma=sigma_live,
              observed=np.log(avg_data.values))
    avg_dead = death_data[['rep1', 'rep2', 'rep3']].mean(axis=1)
    pm.Normal("Y_dead", mu=pm.math.log(pm.math.clip(dead_solution, 1e-8, np.inf)),sigma=sigma_dead,
              observed=np.log(avg_dead.values))
    
    
#model.debug()

# Specify the sampler
sampler = "Slice Sampler"
tune  = 500
draws = 500

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
    posterior_samples["$\delta$ (death rate)"].values,
    posterior_samples["$P_0$ (init. live)"].values,
    posterior_samples["$D_0$ (init. dead)"].values
]).T  # Shape: (n_samples, 5)

n_plot = 200  # Number of samples to simulate for plotting

plt.figure(figsize=(12, 6))

for i in range(n_plot):
    theta = posterior_array[i]
    # last two are initial conditions
    y0 = [theta[-2], theta[-1]]
    time_finer = np.linspace(data['time (hours)'].values[0], data['time (hours)'].values[-1], 100)
    sol = odeint(cells_growth_death, y0, t=time_finer, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
    live = sol[:, 0]
    dead = sol[:, 1]
    total = live + dead

    plt.subplot(1, 2, 1)
    plt.plot(time_finer, total, '-', color='gray', alpha=0.1)

    plt.subplot(1, 2, 2)
    plt.plot(time_finer, dead, '-', color='gray', alpha=0.1)

# Add data points on top
plt.subplot(1, 2, 1)
plt.plot(data['time (hours)'], data['rep1'], 'o', color='orange')
plt.plot(data['time (hours)'], data['rep2'], 'o', color='blue')
plt.plot(data['time (hours)'], data['rep3'], 'o', color='green')
plt.xlabel('Time (hrs)')
plt.ylabel('Total cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)

plt.subplot(1, 2, 2)
plt.plot(death_data['time (hours)'], death_data['rep1'], 'o', color='orange')
plt.plot(death_data['time (hours)'], death_data['rep2'], 'o', color='blue')
plt.plot(death_data['time (hours)'], death_data['rep3'], 'o', color='green')
#plt.xlabel(death_data.columns[0])
plt.xlabel('Time (hrs)')
plt.ylabel('Dead cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)

plt.tight_layout()
plt.show()

# Save chain to a CSV file
df_trace = az.convert_to_inference_data(obj=trace).to_dataframe(include_coords=False,groups='posterior')
df_trace.to_csv('./../res/logistic_growth_death_chain.csv', index=False)


