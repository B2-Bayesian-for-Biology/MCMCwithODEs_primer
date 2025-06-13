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
 
total_init = (data['rep1'].iloc[0] + data['rep2'].iloc[0] + data['rep3'].iloc[0]) / 3
dead_init = (death_data['rep1'].iloc[0] + death_data['rep2'].iloc[0] + death_data['rep3'].iloc[0]) / 3
live_init = total_init - dead_init

# Initial conditions with correct indexing
y0_guess = [500,live_init,dead_init]  # Initial conditions for live and dead cells
print(f"init condi {y0_guess}")


## differential equation and solvers
def cells_growth_death(y,t,mu_max,Ks,Qn,delta):
    N, P, D = y
    dydt = [0,0,0]
    dydt[0] = -Qn*mu_max*N*P*1e6/(N+Ks)
    dydt[1] = mu_max*N*P/(N+Ks) - delta*P
    dydt[2] = delta*P
    return dydt

def solved_num_cells(y0,t,mu_max,Ks,Qn,delta):
    sol = odeint(cells_growth_death, y0, t, args=(mu_max,Ks,Qn,delta))
    return sol

# Initial guesses for parameters
delta_guess = 0.1
mu_max_guess = 0.3
Ks_guess = 1
Qn_guess = 1e-10


avg_data = data[['rep1', 'rep2', 'rep3']].mean(axis=1)
print(np.log(avg_data))

#initial conditions
t = np.linspace(0, 20, 100)

# guess solution
solution = solved_num_cells(y0_guess,t, mu_max_guess,Ks_guess, Qn_guess, delta_guess)

total_cells = solution[:,1] + solution[:,2]
dead_cells = solution[:,2]
live_cells = solution[:,1]  


plt.subplot(1, 4, 1)
plt.plot(t,solution[:,0],'-',color ='k')
plt.xlabel('Time (hrs)')
plt.xlim(0, 20)

plt.subplot(1, 4, 2)
plt.plot(data['time (hours)'],data['rep1'],'o',color ='orange')
plt.plot(data['time (hours)'],data['rep2'],'o',color ='blue')
plt.plot(data['time (hours)'],data['rep3'],'o',color ='green')
plt.plot(t,total_cells,'-',color ='k')
plt.xlabel('Time (hrs)')
plt.ylabel('Total cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)


plt.subplot(1, 4, 3)
plt.plot(death_data['time (hours)'],death_data['rep1'],'o',color ='orange')
plt.plot(death_data['time (hours)'],death_data['rep2'],'o',color ='blue')
plt.plot(death_data['time (hours)'],death_data['rep3'],'o',color ='green')
plt.plot(t,dead_cells,'-',color ='k')
plt.xlabel('Dead Cells')
plt.ylabel('Cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)



plt.subplot(1, 4, 4)
live_data = data['rep1'].values - death_data['rep1'].values
plt.plot(data['time (hours)'],live_data,'o',color ='orange')
plt.plot(t,live_cells,'-',color ='k')
plt.xlabel('Live Cells')
plt.ylabel('Cells')
plt.yscale('log')
plt.xlim(0, 20)
plt.show()




# ---------------------------
# INFERENCE
# ---------------------------

@njit
def cells_growth_death(y,t,theta):
    N, P, D = y
    # unpack parameters
    mu_max,Ks,Qn,delta = theta 
    dydt = [0,0,0]
    dydt[0] = -Qn*mu_max*N*P*1e6/(N+Ks)
    dydt[1] = mu_max*N*P/(N+Ks) - delta*P
    dydt[2] = delta*P
    return dydt



@as_op(itypes=[pt.dvector], otypes=[pt.dmatrix])
def pytensor_forward_model_matrix_vary_init(theta):
    y0 = [theta[-3], theta[-2], theta[-1]]  # Initial conditions for live and dead cells
    # Simulate ODE for each time series separately
    result = odeint(cells_growth_death, y0, t=data['time (hours)'].values, args=(theta[:-3],))
    return result



import pytensor.tensor as pt


with pm.Model() as model:
    # Priors parameters
    mu_max = pm.Uniform(r"$\mu_{max}$",lower=0.25, upper=0.35)  
    Ks = pm.Uniform(r"$K_s$" , lower=0.9, upper=1.1)
    Qn = pm.Uniform(r"$Q_n$ (nutrient uptake rate)", lower=0.05e-10, upper=0.15e-10)
    delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.05, upper=0.15)

    
    
    # prior initial conditions
    N0 = pm.Uniform(r"$N_0$ (nutrients)", lower=40, upper=60)  
    P0 = pm.Uniform(r"$P_0$ (init. live)", lower=3.5e5, upper=7e5)
    D0 = pm.Uniform(r"$D_0$ (init. dead)",  lower=1e5, upper=2.8e5)
    
    # prior noise parameters
    sigma_live = pm.HalfNormal(r"$\sigma_L$", 1)
    sigma_dead = pm.HalfNormal(r"$\sigma_D$", 1)

    theta = pm.math.stack([mu_max,Ks,Qn,delta, N0, P0, D0])
    results = pytensor_forward_model_matrix_vary_init(theta)

    live_solution = results[:,1]
    dead_solution = results[:,2]

    total_solution = live_solution + dead_solution
    print(total_solution.shape)

    # Log likelihoods
    #### I need to clip the values to avoid log(0) ####
    total_data = pm.Data("total_cells", np.log(data[['rep1', 'rep2', 'rep3']].mean(axis=1).values)) 
    dead_data = pm.Data("dead_cells", np.log(death_data[['rep1', 'rep2', 'rep3']].mean(axis=1).values)) 
    
    pm.Normal("Y_live", mu=pm.math.log(pm.math.clip(total_solution, 1e-8, np.inf)),sigma=sigma_live,
            observed= total_data)

    pm.Normal("Y_dead", mu=pm.math.log(pm.math.clip(dead_solution, 1e-8, np.inf)),sigma=sigma_dead,
            observed=dead_data)

    
    
    
    
    
#model.debug()

# Specify the sampler
sampler = "Slice Sampler"
tune  = 1000
draws = 1000

# Variable list to give to the sample step parameter
vars_list = list(model.values_to_rvs.keys())[:-1]
print(vars_list)

with model:
    trace = pm.sample(tune=tune, draws=draws,target_accept=0.8, chains=4, cores=4)


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


'''
'''

# Get posterior samples as a NumPy array
posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
posterior_array = np.vstack([
    posterior_samples["$\mu_{max}$"].values,
    posterior_samples["$K_s$"].values,
    posterior_samples["$Q_n$ (nutrient uptake rate)"].values,
    posterior_samples["$\delta$ (death rate)"].values,
    posterior_samples["$N_0$ (nutrients)"].values,
    posterior_samples["$P_0$ (init. live)"].values,
    posterior_samples["$D_0$ (init. dead)"].values
]).T  # Shape: (n_samples, 5)

n_plot = 200  # Number of samples to simulate for plotting

plt.figure(figsize=(12, 6))

for i in range(n_plot):
    theta = posterior_array[i]
    # last two are initial conditions
    y0 = [theta[-3], theta[-2], theta[-1]]
    sol_both = odeint(cells_growth_death, y0, t=data['time (hours)'].values, args=(theta[:-3],))

    live_solution = sol_both[:,1]
    dead_solution = sol_both[:,2]

    total_solution = live_solution + dead_solution

    plt.subplot(1, 2, 1)
    plt.plot(data['time (hours)'], total_solution, '-', color='deepskyblue', alpha=0.1)

    plt.subplot(1, 2, 2)
    plt.plot(data['time (hours)'], dead_solution, '-', color='deepskyblue', alpha=0.1)
    

# Add data points on top
plt.subplot(1, 2, 1)
plt.plot(data['time (hours)'], data['rep1'], 'o', color='black')
plt.plot(data['time (hours)'], data['rep2'], 'o', color='black')
plt.plot(data['time (hours)'], data['rep3'], 'o', color='black')
plt.xlabel('Time (hrs)')
plt.ylabel('Live cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)

plt.subplot(1, 2, 2)
plt.plot(death_data['time (hours)'], death_data['rep1'], 'o', color='black')
plt.plot(death_data['time (hours)'], death_data['rep2'], 'o', color='black')
plt.plot(death_data['time (hours)'], death_data['rep3'], 'o', color='black')
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
# Save trace to a NetCDF file
file_path = './../res/general_case_trace_v0.nc'
az.to_netcdf(trace, file_path)

