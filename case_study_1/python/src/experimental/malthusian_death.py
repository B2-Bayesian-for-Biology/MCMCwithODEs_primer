import numpy as np
import pandas as pd
from scipy.integrate import odeint
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import arviz as az

# Define the differential equation system
def cells_ode(y, t, params):
    mum,delta = params[0],params[1]
    return [(mum-delta) * y[0]]

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
        mum = pm.Normal('mum', mu=0.5, sigma=0.3)
        delta = pm.Normal('delta', mu=0.5, sigma=0.3)
        N0 = pm.Lognormal('N0', mu=np.log(obs[0]), sigma=0.1)
        sigma = pm.HalfNormal("sigma", 1)

        y_hat = cell_model(y0=[N0], theta=[mum,delta])
        pm.Normal("Y_obs", mu=pm.math.log(y_hat[:, 0]), sigma=sigma, observed=np.log(obs))

    return model


# Run inference and return the trace (InferenceData)
def run_inference(model, draws=2000, tune=2000, chains=4):
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains, return_inferencedata=True, target_accept=0.95)
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
    az.to_netcdf(trace, '../../data/death_posterior_trace.nc')

