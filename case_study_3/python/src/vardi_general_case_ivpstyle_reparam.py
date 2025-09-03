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
from utils.plot_utils_v2 import plot_trace, plot_convergence, plot_posterior_pairs, posterior_dynamics, posterior_dynamics_solve_ivp, posterior_dynamics_solve_ivp_flexible # because __init__.py already re-exports it


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





def general_case(t, y, params):
    N, P, D = y
    mu_max = params[0]
    Ks = params[1]
    Qn = params[2]
    delta = params[3]

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
        delta = pm.Uniform("delta", 0.01, 0.09)

        
        N0 = pm.Deterministic("N0", 1000 + ((500 / 1.8e-10) * (Qn - 3.2e-10)))
        P0 = pm.LogNormal("P0", mu=12.2175, sigma=0.1)
        D0 = pm.LogNormal("D0", mu=10.2804, sigma=0.1)

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

    file_path = '../res/vardi_general_chain_reparam.nc'
    # Build and run model
    model = build_pymc_model(ehux_total_time, ehux_total_density, ehux_dead_density)
    


    # Default to False if not defined
    run_inference_flag = False
    plot_trace_flag = False
    plot_convergence_flag = False
    plot_posterior_pairs_flag = False
    plot_dynamics_flag = True


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
    

    if True:
        plot_trace(
        trace=trace,
        model=model,
        fontname='Arial',
        fontsize=12,
        num_prior_samples=2000,
        var_names_map={'mu_max': 'Maximum Growth Rate μ (/day)', 'delta': 'Death Rate δ (/day)', 'Qn': 'Nutrient Quota Qn (ml/cell)', 'P0': 'Initial Live Density (/ml)', 'D0': 'Initial Dead Density (/ml)','sigma_live': 'σ for live cells', 'sigma_dead': 'σ for dead cells'},
        var_order=['mu_max','delta','Qn','P0','D0','sigma_live','sigma_dead'],
        save_path='../figures/vardi_general_chains_reparam.png'
        )
    
    
    if plot_posterior_pairs_flag:
        plot_posterior_pairs(
        trace,
        plot_kind="kde",
        fontname="Arial",
        fontsize=10,
        figsize=(20, 10),
        var_names_map={'mu_max': 'Maximum Growth Rate μ (/day)', 'delta': 'Death Rate δ (/day)', 'Qn': 'Nutrient Quota Qn (ml/cell)', 'P0': 'Initial Live Density (/ml)', 'D0': 'Initial Dead Density (/ml)','sigma_live': 'σ for live cells', 'sigma_dead': 'σ for dead cells'},
        var_order=['mu_max','delta','Qn','P0','D0','sigma_live','sigma_dead'],
        hspace=0.5,
        wspace=0.2,
        save_path='../figures/vardi_general_posterior_reparam.png'
        )
   
    
    if plot_convergence_flag:
        plot_convergence(
            trace,
            fontname="Arial",
            fontsize=15,
            max_lag=80,
            show_geweke=False,
            hspace=0.8,
            thin = 50,
            combine_chains = False,
            figsize=(10, 10),
            save_path="../figures/vardi_general_convergence_reparam.png"
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



    if False:

        n_samples = 500
        burn_in = 0

        t_min = min(ehux_total_time.min(), ehux_dead_time.min())
        t_max = max(ehux_total_time.max(), ehux_dead_time.max())
        t_eval = np.linspace(t_min, t_max, 200)

        posterior = trace.posterior.stack(draws=("chain", "draw"))

        # If N0 not sampled, compute from Qn
        if "N0" not in posterior:
            Qn_vals = posterior["Qn"].values
            N0_vals = 1000 + ((500 / 1.8e-10) * (Qn_vals - 3.2e-10))
            import xarray as xr
            posterior["N0"] = xr.DataArray(N0_vals, dims=posterior["Qn"].dims, coords=posterior["Qn"].coords)

        available_draws = posterior.draws.size
        assert burn_in + n_samples <= available_draws, "Not enough samples"

        # 3 subplots: Total, Dead, Nutrient
        fig, axes = plt.subplots (1,3, figsize=(12, 4), sharex=True)
        labels = ["Total cells (/ml)", "Dead cells (/ml)", "Nutrients"]
        colors = ["skyblue", "skyblue", "skyblue"]
        sol_keys = ["total", "dead"]

        for i in range(n_samples):
            idx = burn_in + i

            mu_max = posterior["mu_max"].values.flatten()[idx]
            Ks     = posterior["Ks"].values.flatten()[idx]
            Qn     = posterior["Qn"].values.flatten()[idx]
            delta  = posterior["delta"].values.flatten()[idx]
            N0     = posterior["N0"].values.flatten()[idx]
            P0     = posterior["P0"].values.flatten()[idx]
            D0     = posterior["D0"].values.flatten()[idx]

            theta = [mu_max, Ks, Qn, delta]
            y0 = [N0, P0, D0]

            try:
                sol = solve_ivp(
                    fun=lambda t, y: general_case(t, y, theta),
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=y0,
                    t_eval=t_eval,
                    rtol=1e-6,
                    atol=1e-6
                )

                if not sol.success:
                    continue
                sol_data = ode_solution2data(sol.y.T)

                axes[0].plot(t_eval, sol_data["total"], color=colors[0], alpha=0.05)
                axes[1].plot(t_eval, sol_data["dead"],  color=colors[1], alpha=0.05)
                axes[2].plot(t_eval, sol.y[0],  color=colors[2], alpha=0.05)  # Nutrient N directly


            except Exception as e:
                print(f"[Sample {i}] ODE solve failed: {e}")
                continue

        # Observed data overlays
        axes[0].scatter(ehux_total_time, ehux_total_density, color="black", s=20)
        axes[1].scatter(ehux_dead_time,  ehux_dead_density, color="black", s=20)

        # Titles and formatting
        for i, ax in enumerate(axes):
            ax.set_title(labels[i])
            ax.set_xlabel("Time (days)", fontsize=14)
            ax.set_ylabel("Density" if i < 2 else "mmol N $m^{-3}$", fontsize=14)
            ax.set_yscale('log' if i < 2 else 'linear')
            ax.legend(fontsize=12)
            ax.tick_params(labelsize=12)


        plt.tight_layout()
        plt.show()