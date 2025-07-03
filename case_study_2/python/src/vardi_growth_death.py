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
import os
import sys

# Get path to MCMCwithODEs_primer (3 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils import plot_trace, plot_convergence, plot_posterior_pairs, posterior_dynamics2, posterior_dynamics, posterior_dynamics3  # because __init__.py already re-exports it


######## load data

# load data
dataset = pd.read_csv("./../data/total_cells.csv")

ehux_d7_cells  = dataset.head(15)
ehux_cells = dataset.tail(15)

ehux_total_time = ehux_cells['Time (days)'].values
ehux_total_density = 1e6*ehux_cells[' Density (1e6/ml)'].values
ehux_d7_total_time = ehux_d7_cells['Time (days)'].values
ehux_d7_total_density = 1e6*ehux_d7_cells[' Density (1e6/ml)'].values

death_dataset = pd.read_csv("./../data/death_percentage.csv")

ehux_death = death_dataset.head(15)
ehux_d7_death = death_dataset.tail(15)

ehux_dead_time = ehux_death['Time (days)']
ehux_dead_density = ehux_death[' Dead percentage '].values*ehux_total_density/100
ehux_d7_dead_time = ehux_d7_death['Time (days)']
ehux_d7_dead_density = ehux_d7_death[' Dead percentage ']*ehux_d7_total_density/100

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
    """
    User-defined function to extract and compute useful outputs from the ODE solution.
    
    Args:
        solution (np.ndarray): shape (time_points, num_variables)

    Returns:
        dict: keys are output variable names, values are 1D arrays over time
    """
    live = solution[:, 0]
    dead = solution[:, 1]
    total = live + dead
    return {
        "total": total,
        "dead": dead
    }

# Build and return a PyMC model
def build_pymc_model(ehux_total_time, ehux_total_density,ehux_dead_time, ehux_dead_density ):


    cell_model = pm.ode.DifferentialEquation(
        func=logistic_growth_death,
        times=ehux_total_time,
        n_states=2,
        n_theta=3, # because rest goes in y0 
        t0=0
    )

    with pm.Model() as model:
        # Priors
        r = pm.Uniform(r"$r$ (growth rate)", lower=0.5, upper=1)
        K = pm.Uniform(r"$K$ (carrying capacity)" , lower=1e6, upper=4e7)
        delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.0, upper=0.15)
        P0 = pm.Uniform(r"$P_0$ (init. live)", lower=1e5, upper=3e5)
        D0 = pm.Uniform(r"$D_0$ (init. dead)", lower=2e4, upper=4e4)

        sigma_live = pm.HalfNormal(r"$\sigma_L$", 3)
        sigma_dead = pm.HalfNormal(r"$\sigma_D$", 3)

        # Solve the ODE system
        y_hat = cell_model(y0=[P0,D0], theta=[r,K,delta])
        live_solution = y_hat[:,0]
        dead_solution = y_hat[:,1]
        total_solution = live_solution + dead_solution

        # Log likelihoods
        #### I need to clip the values to avoid log(0) ####
        #total_data = pm.Data("total_cells", np.log(data[['rep1', 'rep2', 'rep3']].mean(axis=1).values)) 
        #dead_data = pm.Data("dead_cells", np.log(death_data[['rep1', 'rep2', 'rep3']].mean(axis=1).values)) 
        
        pm.Normal("Y_live", mu=pm.math.log(pm.math.clip(total_solution, 1e-8, np.inf)),sigma=sigma_live,
                observed =  np.log(ehux_total_density))

        pm.Normal("Y_dead", mu=pm.math.log(pm.math.clip(dead_solution, 1e-8, np.inf)),sigma=sigma_dead,
                observed=np.log(ehux_dead_density))

    return model    


def run_inference(model, draws=500, tune=500, chains=4, cores = 4):
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, return_inferencedata=True, target_accept=0.95, cores = cores)
        #pm.set_data({"total_cells": total_data, "dead_cells": dead_data})
        #posterior_predictive = pm.sample_posterior_predictive(trace)
    
    return trace


if __name__ == "__main__":

    file_path = '../res/vardi_logistic_growth_death_chain.nc'
    # Build and run model
    
    model = build_pymc_model(ehux_total_time, ehux_total_density,ehux_dead_time, ehux_dead_density)
    
    if not os.path.exists(file_path):
        trace = run_inference(model)
        az.to_netcdf(trace, file_path)
    else:
        print(f"{file_path} already exists. Skipping model run.")


    # Plotting part
    trace = az.from_netcdf(file_path)

    '''
    plot_trace(
    trace=trace,
    model=model,
    fontname='Arial',
    fontsize=12,
    num_prior_samples=2000,
    save_path='../figures/vardi_growth_death_chains.png'
    )
    

    plot_posterior_pairs(
    trace,
    plot_kind="kde",
    fontname="Arial",
    fontsize=12,
    figsize=(10, 10),
    hspace=0.5,
    wspace=0.5,
    save_path='../figures/vardi_growth_death_posterior.png'
    )
   
    
   
    plot_convergence(
    trace,
    thin=1,
    fontname="Arial",
    fontsize=15,
    max_lag=80,
    show_geweke=False,
    hspace=0.8,
    combine_chains = False,
    save_path="../figures/vardi_growth_death_convergence.png"
    )
    '''
    

    
    ## dataset
    dataset_postprocessing = {
    "Total cells": [
            {"time": ehux_total_time, "values":  ehux_total_density},  # replicate 1
        ],
    "Dead cells": [
            {"time": ehux_dead_time, "values": ehux_dead_density},  # replicate 1
        ]
    }

    posterior_dynamics3(
    dataset_postprocessing,
    trace=trace,
    model=model,
    n_plots=200,
    burn_in=50,
    num_variables=2,
    ode_fn=logistic_growth_death,
    ode2data_fn=ode_solution2data,
    save_path="../figures/vardi_growth_death_dynamics.png",
    var_properties=None,
    figsize=(10, 4),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None
)

