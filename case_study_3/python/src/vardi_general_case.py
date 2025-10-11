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
from utils.plot_utils_v2 import plot_trace, plot_convergence, plot_posterior_pairs, posterior_dynamics # because __init__.py already re-exports it


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
def general_case(y, t, params):

    # Use indexing instead of unpacking
    N, P, D = y[0], y[1], y[2] 
    mu_max,Ks,Qn,delta = params[0], params[1], params[2], params[3]

    dydt = [0, 0, 0]
    
    # Convert P from cells/mL to cells/m^3
    P_m3 = P * 1e6

    # Growth rate: Monod term
    mu = mu_max * N / (N + Ks)

    # ODEs
    dydt[0] = -Qn * mu * P_m3              # mmol N m^-3 day^-1
    dydt[1] = mu * P - delta * P            # cells/mL/day
    dydt[2] = delta * P                     # cells/mL/day

    return dydt



def ode_solution2data(solution):
    """
    User-defined function to extract and compute useful outputs from the ODE solution.
    
    Args:
        solution (np.ndarray): shape (time_points, num_variables)

    Returns:
        dict: keys are output variable names, values are 1D arrays over time
    """
    
    live = solution[:, 1]
    dead = solution[:, 2]
    total = live + dead
    return {
        "total": total,
        "dead": dead
    }

# Build and return a PyMC model
def build_pymc_model(ehux_total_time, ehux_total_density,ehux_dead_time, ehux_dead_density ):


    cell_model = pm.ode.DifferentialEquation(
        func=general_case,
        times=ehux_total_time,
        n_states=3,
        n_theta=4, # because rest goes in y0 
        t0=0
    )

    with pm.Model() as model:
        # Priors
        mu_max = pm.Uniform(r"$\mu_{max}$",lower=0.4, upper=0.7)  
        Ks = pm.Uniform(r"$K_s$" , lower=0.05, upper=0.2)
        Qn = pm.Uniform(r"$Q_n$ (nutrient uptake rate)", lower=1e-10, upper=7e-10)
        delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.01, upper=0.09)
        
        # prior initial conditions
        N0 = pm.Uniform(r"$N_0$ (nutrients)", lower=500, upper=2000)  
        P0 = pm.LogNormal(r"$P_0$ (init. live)", mu=12.2175, sigma=0.1)
        D0 = pm.LogNormal(r"$D_0$ (init. dead)",  mu=10.2804, sigma=0.1)

        #P0 = pm.Uniform(r"$P_0$ (init. live)", lower=1e5, upper=3e5)
        #D0 = pm.Uniform(r"$D_0$ (init. dead)", lower=1e4, upper=7e4)
        
        # prior noise parameters
        sigma_live = pm.HalfNormal(r"$\sigma_L$", 1)
        sigma_dead = pm.HalfNormal(r"$\sigma_D$", 1)

        
        # Solve the ODE system
        y_hat = cell_model(y0=[N0,P0,D0], theta=[mu_max,Ks,Qn,delta])
        y_hat_sol = ode_solution2data(y_hat)
        # Extract live and dead cell solutions
        total_solution = y_hat_sol['total']
        dead_solution = y_hat_sol['dead']

        # pymc multiplies this itself (or additive after taking Log of Likelihood.)
        
        
        pm.Normal("Y_live", mu=pm.math.log(pm.math.clip(total_solution, 1e-8, np.inf)),sigma=sigma_live,
                observed =  np.log(ehux_total_density))

        pm.Normal("Y_dead", mu=pm.math.log(pm.math.clip(dead_solution, 1e-8, np.inf)),sigma=sigma_dead,
               observed=np.log(ehux_dead_density))

    return model    

def run_inference(model, draws=500, tune=500, chains=2, cores=2, threshold_for_slice=20,target_accept=0.8):
    with model:
        # Count number of continuous variables (excluding transformed ones)
        num_params = len(model.free_RVs)
        #var_names = [v.name for v in model.free_RVs]
        
        if num_params > threshold_for_slice:
            print(f"Using Slice sampler (parameters: {num_params})")
            step = pm.Slice()
        else:
            print(f"Using NUTS sampler (parameters: {num_params})")
            step = pm.NUTS(target_accept=target_accept)
        
        trace = pm.sample(draws=draws, tune=tune, chains=chains, step=step,
                          return_inferencedata=True, cores=cores)

    return trace

if __name__ == "__main__":

    file_path = '../res/vardi_general_chain.nc'
    # Build and run model
    
    model = build_pymc_model(ehux_total_time, ehux_total_density,ehux_dead_time, ehux_dead_density)
    

    # Default to False if not defined
    run_inference_flag = False
    plot_trace_flag = True
    plot_convergence_flag = False
    plot_posterior_pairs_flag = False
    plot_dynamics_flag = False



    try:
        run_inference_flag
    except NameError:
        run_inference_flag = False
    '''
    if not os.path.exists(file_path) or run_inference_flag:
        print("Running inference...")
        trace = run_inference(model)
        az.to_netcdf(trace, file_path)
    else:
    '''
    print(f"{file_path} already exists. Skipping model run.")
    trace = az.from_netcdf(file_path)
    

    if plot_trace_flag:
        plot_trace(
        trace=trace,
        model=model,
        fontname='Arial',
        fontsize=12,
        num_prior_samples=2000,
        save_path='../figures/vardi_general_chains.png'
        )
    
    
    if plot_posterior_pairs_flag:
        plot_posterior_pairs(
        trace,
        plot_kind="kde",
        fontname="Arial",
        fontsize=10,
        figsize=(20, 10),
        hspace=0.5,
        wspace=0.2,
        save_path='../figures/vardi_general_posterior.png'
        )
   
    
    if plot_convergence_flag:
        plot_convergence(
            trace,
            thin=1,
            fontname="Arial",
            fontsize=15,
            max_lag=80,
            show_geweke=False,
            hspace=0.8,
            combine_chains = False,
            figsize=(10, 10),
            save_path="../figures/vardi_general_convergence.png"
            )
        
    

    
    ## dataset
    dataset_postprocessing = {
    "Total cells": [
            {"time": ehux_total_time, "values":  ehux_total_density},  # replicate 1
        ],
    "Dead cells": [
            {"time": ehux_dead_time, "values": ehux_dead_density},  # replicate 1
        ]
    }



    if plot_dynamics_flag:
        posterior_dynamics(
        dataset=dataset_postprocessing,
        trace=trace,
        model=model,
        n_plots=100,
        burn_in=50,
        num_variables=3, # including latent
        num_sigma=2,
        ode_fn=general_case,
        ode2data_fn=ode_solution2data,
        save_path="../figures/vardi_general_dynamics.png",
        var_properties={
            "Total cells": {"label": "Total", "color": "black", "ylabel": "Total cell density (/ml)", "sol_key": "total","log": True},
            "Dead cells": {"label": "Dead", "color": "black", "ylabel": "Dead cell density (/ml)", "sol_key": "dead","log": True},
        },
        suptitle="Posterior Predictive Dynamics",
        color_lines='skyblue'
        )
        
   
    