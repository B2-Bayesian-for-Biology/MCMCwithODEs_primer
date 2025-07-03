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
from plotting import *
import sys
# Get path to MCMCwithODEs_primer (3 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils import plot_trace, plot_convergence, plot_posterior_pairs, posterior_dynamics2  # because __init__.py already re-exports it


## differential equation and solvers

def logistic_growth_death(y, t, params):
    # Use indexing instead of unpacking
    P = y[0]
    D = y[1]
    r = params[0]
    K = params[1]
    delta = params[2]

    dydt = [0, 0]
    dydt[0] = r * (1 - P / K) * P - delta * P
    dydt[1] = delta * P
    return dydt

def ode_solution2data(solution):
    live_solution = solution[:,0]
    dead_solution = solution[:,1]
    
    total_cells_solution = live_solution + dead_solution
    return total_cells_solution, dead_solution

# Build and return a PyMC model
def build_pymc_model(dataset):

    data = dataset.head(10)
    death_data = dataset.tail(10)

    cell_model = pm.ode.DifferentialEquation(
        func=logistic_growth_death,
        times=data['time (hours)'].values,
        n_states=2,
        n_theta=3, # because rest goes in y0 
        t0=0
    )

    with pm.Model() as model:
        # Priors
        r = pm.Uniform(r"$r$ (growth rate)", lower=0.25, upper=1)
        K = pm.Uniform(r"$K$ (carrying capacity)" , lower=2e6, upper=1e7)
        delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.05, upper=0.25)
        P0 = pm.Uniform(r"$P_0$ (init. live)", lower=3e5, upper=7e5)
        D0 = pm.Uniform(r"$D_0$ (init. dead)", lower=1e5, upper=4e5)

        sigma_live = pm.HalfNormal(r"$\sigma_L$", 3)
        sigma_dead = pm.HalfNormal(r"$\sigma_D$", 3)

        # Solve the ODE system
        y_hat = cell_model(y0=[P0,D0], theta=[r,K,delta])
        live_solution = y_hat[:,0]
        dead_solution = y_hat[:,1]
        total_solution = live_solution + dead_solution

        # Log likelihoods
        #### I need to clip the values to avoid log(0) ####
        total_data = pm.Data("total_cells", np.log(data[['rep1', 'rep2', 'rep3']].mean(axis=1).values)) 
        dead_data = pm.Data("dead_cells", np.log(death_data[['rep1', 'rep2', 'rep3']].mean(axis=1).values)) 
        
        pm.Normal("Y_live", mu=pm.math.log(pm.math.clip(total_solution, 1e-8, np.inf)),sigma=sigma_live,
                observed= total_data)

        pm.Normal("Y_dead", mu=pm.math.log(pm.math.clip(dead_solution, 1e-8, np.inf)),sigma=sigma_dead,
                observed=dead_data)

    return model       

# ---------------------------
# INFERENCE
# ---------------------------
#        

def run_inference(model, draws=500, tune=500, chains=4, cores = 4):
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, return_inferencedata=True, target_accept=0.95, cores = cores)
        #pm.set_data({"total_cells": total_data, "dead_cells": dead_data})
        #posterior_predictive = pm.sample_posterior_predictive(trace)
    
    return trace


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":

    # load data
    dataset= pd.read_csv("./../data/ehux379_sytox.csv")

    data = dataset.head(10)
    death_data = dataset.tail(10)
    
    file_path = '../data/logistic_growth_death_chain.nc'
    # Build and run model
    model = build_pymc_model(dataset)
    
    if not os.path.exists(file_path):
        trace = run_inference(model)
        az.to_netcdf(trace, file_path)
    else:
        print(f"{file_path} already exists. Skipping model run.")


    # Plotting part
    trace = az.from_netcdf(file_path)


    ## dataset
    dataset_postprocessing = {
    "Total cells": [
            {"time": data['time (hours)'], "values":  data['rep1']},  # replicate 1
            {"time": data['time (hours)'], "values":  data['rep2']},  # replicate 2
            {"time": data['time (hours)'], "values":  data['rep3']},  # replicate 3
        ],
    "Dead cells": [
            {"time": death_data['time (hours)'], "values":  death_data['rep1']},  # replicate 1
            {"time": death_data['time (hours)'], "values":  death_data['rep2']},  # replicate 2
            {"time": death_data['time (hours)'], "values":  death_data['rep3']},  # replicate 3
        ]
    }

    plot_trace(
    trace=trace,
    model=model,
    #var_names_map={'N0':'Initial density/ml','mum': 'Growth Rate μ', 'sigma': 'Std Dev σ'},
    #var_order=['mum','N0','sigma'],
    fontname='Arial',
    fontsize=12,
    num_prior_samples=2000
    #save_path='../figures/normal_growth_chains.png'
    )
    

    plot_posterior_pairs(
    trace,
    #var_names=["mum", "N0", "sigma"],
    #var_names_map={"mum": "Growth Rate μ", "N0": "Initial density/ml", "sigma": "Std Dev σ"},
    #var_order=["mum", "N0", "sigma"],
    plot_kind="kde",
    fontname="Arial",
    fontsize=12,
    figsize=(10, 10),
    hspace=0.5,
    wspace=0.5
    )
   
    
   
    plot_convergence(
    trace,
    #var_names=["mum", "N0", "sigma"],
    #var_order=["mum", "N0", "sigma"],
    #var_names_map={"mum": "Growth Rate μ", "N0": "Initial density/ml", "sigma": "Std Dev σ"},
    thin=1,
    fontname="Arial",
    fontsize=15,
    max_lag=80,
    show_geweke=False,
    hspace=0.8,
    combine_chains = False,
    save_path="../figures/normal_growth_convergence.png"
    )

    
    
    
    
    #plot_trace(trace, save_path='../figures/logistic_growth_death_trace.png')
    #plot_posterior(trace, save_path='../figures/logistic_growth_death_posterior.png')
    #plot_autocorrelation(trace, save_path='../figures/logistic_growth_death_autocorrelation.png')
    #run_posterior_predictive_checks(model, trace, var_names=["Y_live", "Y_dead"], plot=True, savepath='../figures/logistic_growth_death_posterior_predictive')

    
    # Extracting the time and cell counts for plotting

    #df_list = [data, death_data]  # Add more as needed
    #y_vector_data = [df[['rep1', 'rep2', 'rep3']].values for df in df_list]
    

    ############## To be optimized ##############
    
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
        sol = odeint(logistic_growth_death, y0, t=time_finer, args=(theta[:-2],), rtol=1e-6, atol=1e-6)
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

   


'''


posterior_dynamics2(
    dataset_postprocessing,
    model=model,
    time_vec=np.linspace(data['time (hours)'].values[0], data['time (hours)'].values[-1], 100),
    trace=trace,
    ode_fn=logistic_growth_death,
    ode_var_names=["N0", "mum"],
    ode_state_count=1,
    num_lines=100,
    var_properties={
        "Total Cells": {
            "label": "Cells",
            "color": "tab:red",
            "ylabel": "Cells/mL"
        },
        "Dead Cells": {
            "label": "Dead Cells",
            "color": "tab:blue",
            "ylabel": "Cells/mL"
        }
    },
    fontname="Arial",
    fontsize=12,
    line_width=2.5,
    figsize=(8, 3),
    suptitle="Cell Dynamics Over Time"
)

# Posterior predictive checks
with model:
    

# Extracting predictions
live_pred = posterior_predictive['Y_live']
dead_pred = posterior_predictive['Y_dead']

# Plotting posterior predictive
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Live cells
axs[0].plot(data['time (hours)'], np.exp(live_pred).mean(axis=0), label='Posterior Predictive Mean', color='blue')
axs[0].fill_between(data['time (hours)'], 
                    np.percentile(np.exp(live_pred), 5, axis=0), 
                    np.percentile(np.exp(live_pred), 95, axis=0), 
                    color='blue', alpha=0.3, label='95% CI')
axs[0].scatter(data['time (hours)'], np.exp(total_data), color='black', label='Observed Data')
axs[0].set_title('Posterior Predictive for Live Cells')
axs[0].set_xlabel('Time (hours)')
axs[0].set_ylabel('Live Cells')
axs[0].legend()

# Dead cells
axs[1].plot(death_data['time (hours)'], np.exp(dead_pred).mean(axis=0), label='Posterior Predictive Mean', color='red')
axs[1].fill_between(death_data['time (hours)'], 
                    np.percentile(np.exp(dead_pred), 5, axis=0), 
                    np.percentile(np.exp(dead_pred), 95, axis=0), 
                    color='red', alpha=0.3, label='95% CI')
axs[1].scatter(death_data['time (hours)'], np.exp(dead_data), color='black', label='Observed Data')
axs[1].set_title('Posterior Predictive for Dead Cells')
axs[1].set_xlabel('Time (hours)')
axs[1].set_ylabel('Dead Cells')
axs[1].legend()

plt.tight_layout()
plt.savefig('../figures/logistic_growth_death_posterior_predictive.png')
plt.show()

'''
