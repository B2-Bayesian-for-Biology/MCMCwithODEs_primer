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
    return trace


# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":

    # load data
    dataset= pd.read_csv("./../data/ehux379_sytox.csv")

    
    
    file_path = '../data/logistic_growth_death_chain.nc'

    if not os.path.exists(file_path):
        # Build and run model
        model = build_pymc_model(dataset)
        trace = run_inference(model)
        az.to_netcdf(trace, file_path)
    else:
        print(f"{file_path} already exists. Skipping model run.")


    # Plotting part
    trace = az.from_netcdf(file_path)
    plot_trace(trace, save_path='../figures/logistic_growth_death_trace.png')
    
