'''
Cannot use NUTS with solve_ivp
NUTS needs gradients
Use Slice sampler instead or Metropolis.
'''



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


import pytensor
import pytensor.tensor as pt
from pytensor.graph.op import Op
from pytensor.graph.basic import Apply
from scipy.integrate import solve_ivp


# Get path to MCMCwithODEs_primer (3 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils import plot_trace, plot_convergence, plot_posterior_pairs, posterior_dynamics # because __init__.py already re-exports it


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

ehux_death = death_dataset.tail(15)
ehux_d7_death = death_dataset.head(15)

ehux_dead_time = ehux_death['Time (days)']
ehux_dead_density = ehux_death[' Dead percentage '].values*ehux_total_density/100
ehux_d7_dead_time = ehux_d7_death['Time (days)']
ehux_d7_dead_density = ehux_d7_death[' Dead percentage ']*ehux_d7_total_density/100





def general_case(t, y, params):
    N, P, D = y
    mu_max, Ks, Qn, delta = params

    P_m3 = P * 1e6  # cells/mL → cells/m³
    mu = mu_max * N / (N + Ks)

    dNdt = -Qn * mu * P_m3
    dPdt = mu * P - delta * P
    dDdt = delta * P

    return [dNdt, dPdt, dDdt]



class SolveIVPWrapper(Op):
    itypes = [pt.dvector]  # theta + y0
    otypes = [pt.dmatrix]  # solution: (len(t), 3)

    def __init__(self, times):
        self.times = times

    def perform(self, node, inputs, outputs):
        theta_y0, = inputs
        theta = theta_y0[:4]
        y0 = theta_y0[4:]

        sol = solve_ivp(
            fun=lambda t, y: general_case(t, y, theta),
            t_span=(self.times[0], self.times[-1]),
            y0=y0,
            t_eval=self.times,
            method="LSODA"
        )

        if not sol.success:
            raise RuntimeError("ODE solver failed:", sol.message)

        outputs[0][0] = sol.y.T  # shape: (time, 3)




# === Convert solution to observables ===
def ode_solution2data(solution):
    live = solution[:, 1]
    dead = solution[:, 2]
    total = live + dead
    return {
        "total": total,
        "dead": dead
    }



def build_pymc_model(times, total_obs, dead_obs):
    ode_op = SolveIVPWrapper(times)

    with pm.Model() as model:
        mu_max = pm.Uniform("mu_max", 0.4, 0.7)
        Ks = pm.Uniform("Ks", 0.05, 0.2)
        Qn = pm.Uniform("Qn", 1e-10, 7e-10)

        delta = pm.Normal("delta", mu = 0.034, sigma = 0.003)
        #delta = pm.Uniform("delta", 0.01, 0.09)

        N0 = pm.Uniform("N0", 500, 2000)
        #N0 = pm.Deterministic("N0", 1000 + ((500 / 1.8e-10) * (Qn - 3.2e-10)))
        P0 = pm.Normal("P0", mu=211017, sigma=14380)  # Based on data, around
        D0 = pm.Normal("D0", mu=31104, sigma=5761)  # Based on data, around

        sigma_live = pm.HalfNormal("sigma_live", 1)
        sigma_dead = pm.HalfNormal("sigma_dead", 1)

        # Solve ODE
        sol = ode_op(pt.stack([mu_max, Ks, Qn, delta, N0, P0, D0]))

        total = sol[:, 1] + sol[:, 2]
        dead = sol[:, 2]

        pm.Normal("Y_total", mu=pt.log(total), sigma=sigma_live, observed=np.log(total_obs))
        pm.Normal("Y_dead", mu=pt.log(dead), sigma=sigma_dead, observed=np.log(dead_obs))

    return model


def run_inference(model, draws=10000, tune=5000, chains=3, cores=3, threshold_for_slice=5,target_accept=0.9):
    with model:
        # Count number of continuous variables (excluding transformed ones)
        num_params = len(model.free_RVs)
        #var_names = [v.name for v in model.free_RVs]
        
        if num_params > threshold_for_slice:
            print(f"Using Slice sampler (parameters: {num_params})")
            #step = pm.Slice()
            step = pm.Metropolis()
        else:
            print(f"Using NUTS sampler (parameters: {num_params})")
            step = pm.NUTS(target_accept=target_accept)
        
        trace = pm.sample(draws=draws, tune=tune, chains=chains, step=step,
                          return_inferencedata=True, cores=cores)

    return trace







if __name__ == "__main__":

    file_path = '../res/vardi_general_chain_corr.nc'
    # Build and run model
    model = build_pymc_model(ehux_total_time, ehux_total_density, ehux_dead_density)
    


    # Default to False if not defined
    run_inference_flag = True
    plot_trace_flag = True
    plot_convergence_flag = True
    plot_posterior_pairs_flag = True
    plot_dynamics_flag = False


    try:
        run_inference_flag
    except NameError:
        run_inference_flag = False

    if not os.path.exists(file_path) or run_inference_flag:
        print("Running inference...")
        trace = run_inference(model)
        az.to_netcdf(trace, file_path)
    else:
        print(f"{file_path} already exists. Skipping model run.")
        trace = az.from_netcdf(file_path)
    

    if plot_trace_flag:
        plot_trace(
        trace=trace,
        model=model,
        fontname='Arial',
        fontsize=12,
        num_prior_samples=2000,
        save_path='../figures/vardi_general_chains_corr.png'
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
        save_path='../figures/vardi_general_posterior_corr.png'
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
            save_path="../figures/vardi_general_convergence_corr.png"
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
        save_path="../figures/vardi_general_dynamics_corr.png",
        var_properties={
            "Total cells": {"label": "Total", "color": "black", "ylabel": "Total cell density (/ml)", "sol_key": "total","log": True},
            "Dead cells": {"label": "Dead", "color": "black", "ylabel": "Dead cell density (/ml)", "sol_key": "dead","log": True},
        },
        suptitle="Posterior Predictive Dynamics",
        color_lines='skyblue'
        )
        
   
    