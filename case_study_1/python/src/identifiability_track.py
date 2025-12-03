import numpy as np
import pandas as pd
from scipy.integrate import odeint
import pymc as pm
from pytensor.compile.ops import as_op
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

# Define the differential equation system
def cells_ode(y, t, params):
    mum,delta = params[0],params[1]
    return [(mum-delta) * y[0]]

# Build and return a PyMC model
def build_pymc_model(time, ob, sigma_delta):

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
        delta = pm.TruncatedNormal('delta', mu=0.2, sigma=sigma_delta, lower=0.01, upper=1.0)
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
    data = pd.read_csv("./../data/phaeocystis_control.csv")
    time = data['times'].values
    obs = data['cells'].values

    
    # Sensitivity analysis for different sigma_delta values
    sigma_delta_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    mum_stddevs = []

    for sd in sigma_delta_values:
        print(f"\nRunning inference for sigma_delta = {sd}")
        model = build_pymc_model(time, obs, sigma_delta=sd)
        trace = run_inference(model, draws=500, tune=500, chains=2)  # Faster runs for test
        std_mum = trace.posterior['mum'].std().values.flatten()[0]
        mum_stddevs.append(std_mum)
        print(f"Posterior std of mum: {std_mum:.4f}")

        # save trace files
        az.to_netcdf(trace, "./../data/trace_sigma_delta_" + str(sd) + ".nc")
        
    # Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_delta_values, mum_stddevs, marker='o', linestyle='none', color='red')
    plt.xlabel("Prior Std Dev of $\delta$")
    plt.ylabel("Posterior Std Dev of $\mu$")
    plt.title("Effect of Prior Uncertainty in $\delta$ on Posterior of $\mu$")
    plt.tick_params(labelsize=18)
    plt.xlabel("Prior Std Dev of $\delta$", fontsize=20)
    plt.ylabel("Posterior Std Dev of $\mu$", fontsize=20)
    plt.title("Effect of Prior Uncertainty in $\delta$ on Posterior of $\mu$", fontsize=20)
    #plt.grid(True)
    plt.tight_layout()
    plt.show()

