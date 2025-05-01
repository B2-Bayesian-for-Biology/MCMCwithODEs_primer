import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pymc as pm
import arviz as az
from pytensor.compile.ops import as_op
import pytensor.tensor as pt

# ---------------------------
# MODELING
# ---------------------------

# Define the differential equation for exponential cell growth
def cells_ode(y, t, mum):
    return mum * y

# Solve the ODE numerically given initial conditions, time, and parameter
def solve_cells_ode(y0, t, mum):
    return odeint(cells_ode, y0, t, args=(mum,)).flatten()

# Define a PyMC-compatible wrapper for ODE solving
@as_op(itypes=[pt.dscalar, pt.dscalar], otypes=[pt.dvector])
def pymc_cells_model(mum, N0):
    return solve_cells_ode(N0, pymc_cells_model.time_data, mum)

# ---------------------------
# INFERENCE
# ---------------------------

# Build and return a PyMC model
def build_pymc_model(time, obs):
    pymc_cells_model.time_data = time  # set global state for @as_op function

    with pm.Model() as model:
        # Priors
        mum = pm.Normal('mum', mu=0.5, sigma=0.3)
        N0 = pm.Lognormal('N0', mu=np.log(obs[0]), sigma=0.1)
        sigma = pm.HalfNormal("sigma", 1)

        y_hat = pymc_cells_model(mum, N0)
        pm.Normal("Y_obs", mu=np.log(y_hat), sigma=sigma, observed=np.log(obs))

    return model

# Run inference and return the trace (InferenceData)
def run_inference(model, draws=1000, tune=1000, chains=2):
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
    trace = run_inference(model)

    # save trace plots to csv
    az.to_netcdf(trace, '../data/posterior_trace.nc')

