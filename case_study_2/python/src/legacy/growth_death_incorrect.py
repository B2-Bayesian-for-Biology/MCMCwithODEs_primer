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


data = pd.read_csv("./../data/phaeocystis_control.csv")
death_data = pd.read_csv("./../data/ehux379_sytox.csv")
death_data = death_data.head(10)




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

r_guess = 0.7
K_guess = 5e6
delta_guess = 0.1




#initial conditions
y0_guess = [data['cells'][0],(death_data['rep1'][0]+ death_data['rep2'][0]+death_data['rep3'][0])/3]
t = np.linspace(0, 20, 100)

# guess solution
solution = solved_num_cells(y0_guess,t, r_guess, K_guess, delta_guess)


plt.subplot(1, 2, 1)
plt.plot(data['times'],data['cells'],'o',color ='orange')
plt.plot(t,solution[:,0],'-',color ='k')
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
    result_1 = odeint(cells_growth_death, y0, t=data['times'].values, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
    result_2 = odeint(cells_growth_death, y0, t=death_data['time (hours)'].values, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
    return result_1, result_2



import pytensor.tensor as pt


with pm.Model() as model:
    # Priors
    r = pm.Uniform('r', lower=0.1, upper=0.9)
    K = pm.Uniform('K', lower=1e7, upper=1e9)
    delta = pm.Uniform('delta', lower=0.05, upper=0.2)
    P0 = pm.Uniform('P0', lower=1e6, upper=5e7)
    D0 = pm.Uniform('D0', lower=5e5, upper=2e6)

    sigma_live = pm.HalfNormal("sigma_live", 10)
    sigma_dead = pm.HalfNormal("sigma_dead", 10)

    theta = pm.math.stack([r, K, delta, P0, D0])
    live_solution, dead_solution = pytensor_forward_model_matrix_vary_init(theta)

    # Likelihoods
    pm.Normal("Y_live", mu=pm.math.log(live_solution[:, 0]), sigma=sigma_live,
              observed=np.log(data['cells'].values))
    
    avg_dead = death_data[['rep1', 'rep2', 'rep3']].mean(axis=1)
    pm.Normal("Y_dead", mu=pm.math.log(dead_solution[:, 1]), sigma=sigma_dead,
              observed=np.log(avg_dead.values))
    


# Specify the sampler
sampler = "Slice Sampler"
tune = draws = 3000


# Variable list to give to the sample step parameter
vars_list = list(model.values_to_rvs.keys())[:-1]
print(vars_list)

with model:
    trace = pm.sample(tune=tune, draws=draws)


# Plot chains
az.plot_trace(trace)
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
    posterior_samples['r'].values,
    posterior_samples['K'].values,
    posterior_samples['delta'].values,
    posterior_samples['P0'].values,
    posterior_samples['D0'].values
]).T  # Shape: (n_samples, 5)

n_plot = 200  # Number of samples to simulate for plotting

plt.figure(figsize=(12, 6))

for i in range(n_plot):
    theta = posterior_array[i]
    # last two are initial conditions
    y0 = [theta[-2], theta[-1]]
    sol_live = odeint(cells_growth_death, y0, t=data['times'].values, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
    sol_dead = odeint(cells_growth_death, y0, t=death_data['time (hours)'].values, args=(theta[:-2],), rtol=1e-6, atol=1e-6)

    plt.subplot(1, 2, 1)
    plt.plot(data['times'], sol_live[:, 0], '-', color='gray', alpha=0.1)

    plt.subplot(1, 2, 2)
    plt.plot(death_data['time (hours)'], sol_dead[:, 1], '-', color='gray', alpha=0.1)

# Add data points on top
plt.subplot(1, 2, 1)
plt.plot(data['times'], data['cells'], 'o', color='orange')
plt.xlabel('Time (hrs)')
plt.ylabel('Live cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)

plt.subplot(1, 2, 2)
plt.plot(death_data['time (hours)'], death_data['rep1'], 'o', color='orange')
plt.plot(death_data['time (hours)'], death_data['rep2'], 'o', color='blue')
plt.plot(death_data['time (hours)'], death_data['rep3'], 'o', color='green')
plt.xlabel(death_data.columns[0])
plt.ylabel('Dead cells')
plt.yscale('log')
plt.ylim(1e5, 1e7)
plt.xlim(0, 20)

plt.tight_layout()
plt.show()