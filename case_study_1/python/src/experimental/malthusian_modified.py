import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pymc as pm
import arviz as az
from pytensor.compile.ops import as_op
import pytensor.tensor as pt


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
    data = pd.read_csv("./../../data/phaeocystis_control.csv")
    time = data['times'].values
    obs = data['cells'].values

    # Build and run model
    model = build_pymc_model(time, obs)
    trace = run_inference(model)

    # save trace plots to csv
    az.to_netcdf(trace, '../../data/posterior_trace.nc')

