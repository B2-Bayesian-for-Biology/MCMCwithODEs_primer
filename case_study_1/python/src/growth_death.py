import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pymc as pm
import arviz as az
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import os
import sys
# Get path to MCMCwithODEs_primer (3 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from utils import plot_trace, plot_convergence, plot_posterior_pairs, posterior_dynamics  # because __init__.py already re-exports it

# Define the differential equation system
# Define the differential equation system
def cells_ode(y, t, params):
    mum,delta = params[0],params[1]
    return [(mum-delta) * y[0]]

# Define after solving the equation, how to define variables to compare to data
def ode_solution2data(solution):
    """
    User-defined function to extract and compute useful outputs from the ODE solution.
    
    Args:
        solution (np.ndarray): shape (time_points, num_variables)

    Returns:
        dict: keys are output variable names, values are 1D arrays over time
    """
    
    total = solution[:, 0]  # Assuming the first column is the total cell count
    return {
        "total": total
    }


# Load data
data = pd.read_csv("./../data/phaeocystis_control.csv")
time = data['times'].values
obs = data['cells'].values


# Build and return a PyMC model
def build_pymc_model(time, obs):

    cell_model = pm.ode.DifferentialEquation(
        func=cells_ode,
        times=data['times'].values,
        n_states=1,
        n_theta=2,
        t0=0
    )



    with pm.Model() as model:
        # Priors
        mum = pm.TruncatedNormal('mum', mu=0.5, sigma=0.3,lower = 0.1, upper=1.0)
        delta = pm.TruncatedNormal('delta', mu=0.5, sigma=0.3, lower=0.1, upper=1.0)
        N0 = pm.Lognormal('N0', mu=np.log(obs[0]), sigma=0.1)
        sigma = pm.HalfNormal("sigma", 1)

        y_hat = cell_model(y0=[N0], theta=[mum,delta])
        pm.Normal("Y_obs", mu=pm.math.log(y_hat[:, 0]), sigma=sigma, observed=np.log(obs))

    return model


# Run inference and return the trace (InferenceData)
def run_inference(model, draws=1000, tune=1000, chains=4):
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, return_inferencedata=True)
    return trace

# ---------------------------
# MAIN EXECUTION
# ---------------------------

# This block only runs if the script is executed directly (not when imported)
if __name__ == "__main__":
    

    # Build and run model
    model = build_pymc_model(time, obs)
    # Default to False if not defined
    run_inference_flag = False
    plot_trace_flag = True
    plot_convergence_flag = True
    plot_posterior_pairs_flag = True
    plot_dynamics_flag = True

    file_path = '../data/normal_death_trace.nc'
    
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


    

    ## dataset
    dataset_postprocessing = {
    "Cells": [
            {"time": time, "values": obs},  # replicate 1
        ]
    }

    if plot_trace:
        plot_trace(
        trace=trace,
        model=model,
        var_names_map={'N0':'Initial density/ml','mum': 'Growth Rate μ (/day)', 'delta': 'Death Rate δ (/day)','sigma': 'Std Dev σ'},
        var_order=['mum','delta','N0','sigma'],
        fontname='Arial',
        fontsize=12,
        num_prior_samples=2000,
        save_path='../figures/normal_growthdeath_chains.png'
        )
    
    if plot_posterior_pairs_flag:
        plot_posterior_pairs(
        trace,
        var_names=["mum", "delta", "N0", "sigma"],
        var_names_map={'N0':'Initial density/ml','mum': 'Growth Rate μ (/day)', 'delta': 'Death Rate δ (/day)','sigma': 'Std Dev σ'},
        var_order=['mum','delta','N0','sigma'],
        plot_kind="kde",
        fontname="Arial",
        fontsize=12,
        figsize=(10, 10),
        hspace=0.3,
        wspace=0.3,
        save_path="../figures/normal_growthdeath_posterior.png"
        )
   
    
    if plot_convergence_flag:
        plot_convergence(
        trace,
        var_order=['mum','delta','N0','sigma'],
        var_names=['mum','delta','N0','sigma'],
        var_names_map={'N0':'Initial density/ml','mum': 'Growth Rate μ (/day)', 'delta': 'Death Rate δ (/day)','sigma': 'Std Dev σ'},
        thin=1,
        fontname="Arial",
        fontsize=15,
        max_lag=80,
        show_geweke=False,
        hspace=0.8,
        combine_chains = False,
        save_path="../figures/normal_growthdeath_convergence.png"
        )

    if plot_dynamics_flag:
        posterior_dynamics(
        dataset=dataset_postprocessing,
        trace=trace,
        model=model,
        n_plots=100,
        burn_in=50,
        num_variables=1,
        ode_fn=cells_ode,
        ode2data_fn=ode_solution2data,
        save_path="../figures/vardi_logistic_growthdeath_dynamics.png",
        # the key of this dict is the variable name in the dataset_postprocessing
        var_properties={
            "Cells": {"label": "Total", "color": "black", "ylabel": "Total cell density (/ml)", "xlabel":"Time (days)","sol_key": "total","log": True}
        },
        suptitle="Posterior Predictive Dynamics",
        color_lines='red'
        )

    