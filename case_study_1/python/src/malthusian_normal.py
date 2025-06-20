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


# Get path to MCMCwithODEs_primer (2 levels up)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Now this works:
from utils import plot_trace, plot_convergence, plot_posterior_pairs  # because __init__.py already re-exports it

# Define the differential equation system
def cells_ode(y, t, params):
    mum = params[0]
    return [mum * y[0]]


# Build and return a PyMC model
def build_pymc_model(time, obs):

    cell_model = pm.ode.DifferentialEquation(
        func=cells_ode,
        times=data['times'].values,
        n_states=1,
        n_theta=1,
        t0=0
    )



    with pm.Model() as model:
        # Priors
        mum = pm.Normal('mum', mu=0.5, sigma=0.3)
        N0 = pm.Lognormal('N0', mu=np.log(obs[0]), sigma=0.1)
        sigma = pm.HalfNormal("sigma", 1)

        y_hat = cell_model(y0=[N0], theta=[mum])
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
    # Load data
    data = pd.read_csv("./../data/phaeocystis_control.csv")
    time = data['times'].values
    obs = data['cells'].values

    # Build and run model
    model = build_pymc_model(time, obs)


    file_path = '../data/normal_growth_trace.nc'
    
    if not os.path.exists(file_path):
        print(f"Running model and saving trace to {file_path}")
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
    var_names_map={'N0':'Initial density/ml','mum': 'Growth Rate μ', 'sigma': 'Std Dev σ'},
    var_order=['mum','N0','sigma'],
    fontname='Arial',
    fontsize=12,
    num_prior_samples=2000
    #save_path='../figures/normal_growth_chains.png'
    )
    

    plot_posterior_pairs(
    trace,
    var_names=["mum", "N0", "sigma"],
    var_names_map={"mum": "Growth Rate μ", "N0": "Initial density/ml", "sigma": "Std Dev σ"},
    var_order=["mum", "N0", "sigma"],
    plot_kind="kde",
    fontname="Arial",
    fontsize=12,
    figsize=(10, 10),
    hspace=0.5,
    wspace=0.5
    )
    '''
    
   
    plot_convergence(
    trace,
    var_names=["mum", "N0", "sigma"],
    var_order=["mum", "N0", "sigma"],
    var_names_map={"mum": "Growth Rate μ", "N0": "Initial density/ml", "sigma": "Std Dev σ"},
    thin=1,
    fontname="Arial",
    fontsize=15,
    max_lag=80,
    show_geweke=False,
    hspace=0.8,
    combine_chains = False,
    save_path="../figures/normal_growth_convergence.png"
    )


    #plot_autocorrelation(trace)#, save_path='../figures/general_autocorrelation.png')


