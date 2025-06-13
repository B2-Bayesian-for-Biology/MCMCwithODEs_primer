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
def general_case(y, t, params):

    # Use indexing instead of unpacking
    N, P, D = y[0], y[1], y[2] 
    mu_max,Ks,Qn,delta = params[0], params[1], params[2], params[3]

    dydt = [0, 0, 0]
    

    dydt[0] = -Qn*mu_max*N*P*1e6/(N+Ks)
    dydt[1] = mu_max*N*P/(N+Ks) - delta*P
    dydt[2] = delta*P

    return dydt



# Build and return a PyMC model
def build_pymc_model(dataset):

    data = dataset.head(10)
    death_data = dataset.tail(10)

    cell_model = pm.ode.DifferentialEquation(
        func=general_case,
        times=data['time (hours)'].values,
        n_states=3,
        n_theta=4, # because rest goes in y0 
        t0=0
    )

    with pm.Model() as model:
        '''
        # Priors parameters
        mu_max = pm.Uniform(r"$\mu_{max}$",lower=0.1, upper=0.4)  
        Ks = pm.Uniform(r"$K_s$" , lower=0.05, upper=0.3)
        Qn = pm.Uniform(r"$Q_n$ (nutrient uptake rate)", lower=1e-10, upper=50e-10)
        delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.05, upper=0.3)
        
        # prior initial conditions
        N0 = pm.Uniform(r"$N_0$ (nutrients)", lower=1000, upper=7000)  
        P0 = pm.Uniform(r"$P_0$ (init. live)", lower=3e5, upper=7e5)
        D0 = pm.Uniform(r"$D_0$ (init. dead)",  lower=1e5, upper=4e5)
        
        # prior noise parameters
        sigma_live = pm.HalfNormal(r"$\sigma_L$", 3)
        sigma_dead = pm.HalfNormal(r"$\sigma_D$", 3)
        '''

        mu_max = pm.Uniform(r"$\mu_{max}$",lower=0.2, upper=0.35)  
        Ks = pm.Uniform(r"$K_s$" , lower=0.05, upper=0.3)
        Qn = pm.Uniform(r"$Q_n$ (nutrient uptake rate)", lower=3e-10, upper=7e-10)
        delta = pm.Uniform(r"$\delta$ (death rate)", lower=0.05, upper=0.3)
        
        # prior initial conditions
        N0 = pm.Uniform(r"$N_0$ (nutrients)", lower=4000, upper=8000)  
        P0 = pm.Uniform(r"$P_0$ (init. live)", lower=3e5, upper=7e5)
        D0 = pm.Uniform(r"$D_0$ (init. dead)",  lower=1e5, upper=4e5)
        
        # prior noise parameters
        sigma_live = pm.HalfNormal(r"$\sigma_L$", 3)
        sigma_dead = pm.HalfNormal(r"$\sigma_D$", 3)

        # Solve the ODE system
        y_hat = cell_model(y0=[N0,P0,D0], theta=[mu_max,Ks,Qn,delta])
        live_solution = y_hat[:,1]
        dead_solution = y_hat[:,2]
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


def run_inference(model, draws=1000, tune=1000, chains=4, cores=4):
    with model:
        
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          return_inferencedata=True, target_accept=0.85,
                          cores=cores)
    return trace



# ---------------------------
# MAIN EXECUTION
# ---------------------------

if __name__ == "__main__":

    # load data
    dataset= pd.read_csv("./../data/ehux379_sytox.csv")

    data = dataset.head(10)
    death_data = dataset.tail(10)
    
    file_path = '../res/general_case_chain.nc'
    # Build and run model
    model = build_pymc_model(dataset)
    
    if not os.path.exists(file_path):
        print(f"Running model and saving trace to {file_path}")
        trace = run_inference(model)
        az.to_netcdf(trace, file_path)
    else:
        print(f"{file_path} already exists. Skipping model run.")


    # Plotting part
    trace = az.from_netcdf(file_path)

    #plot_trace(trace, save_path='../figures/general_trace.png')
    #plot_posterior(trace, save_path='../figures/general_posterior.png')
    #plot_autocorrelation(trace, save_path='../figures/general_autocorrelation.png')
    #run_posterior_predictive_checks(model, trace, var_names=["Y_live", "Y_dead"], plot=True, savepath='../figures/general_posterior_predictive')

    
    # Extracting the time and cell counts for plotting

    #df_list = [data, death_data]  # Add more as needed
    #y_vector_data = [df[['rep1', 'rep2', 'rep3']].values for df in df_list]
    

    ############## To be optimized ##############

    # Get posterior samples as a NumPy array
    posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
    posterior_array = np.vstack([
    posterior_samples["$\mu_{max}$"].values,
    posterior_samples["$K_s$"].values,
    posterior_samples["$Q_n$ (nutrient uptake rate)"].values,
    posterior_samples["$\delta$ (death rate)"].values,
    posterior_samples["$N_0$ (nutrients)"].values,
    posterior_samples["$P_0$ (init. live)"].values,
    posterior_samples["$D_0$ (init. dead)"].values
    ]).T  # Shape: (n_samples, 5)

    n_plot = 200  # Number of samples to simulate for plotting

    plt.figure(figsize=(12, 6))

    for i in range(n_plot):
        theta = posterior_array[i]
        # last two are initial conditions
        y0 = [theta[-3], theta[-2], theta[-1]]
        time_finer = np.linspace(data['time (hours)'].values[0], data['time (hours)'].values[-1], 100)
        sol = odeint(general_case, y0, t=time_finer, args=(theta[:-3],), rtol=1e-6, atol=1e-6)
        live = sol[:, 1]
        dead = sol[:, 2]
        total = live + dead

        plt.subplot(1, 2, 1)
        plt.plot(time_finer, total, '-', color='gray', alpha=0.1)

        plt.subplot(1, 2, 2)
        plt.plot(time_finer, dead, '-', color='gray', alpha=0.1)

    # Add data points on top
    plt.subplot(1, 2, 1)
    plt.plot(data['time (hours)'], data['rep1'], 'o', color='black')
    plt.plot(data['time (hours)'], data['rep2'], 'o', color='black')
    plt.plot(data['time (hours)'], data['rep3'], 'o', color='black')
    plt.xlabel('Time (hrs)')
    plt.ylabel('Live cells')
    plt.yscale('log')
    plt.ylim(1e5, 1e7)
    plt.xlim(0, 20)

    plt.subplot(1, 2, 2)
    plt.plot(death_data['time (hours)'], death_data['rep1'], 'o', color='black')
    plt.plot(death_data['time (hours)'], death_data['rep2'], 'o', color='black')
    plt.plot(death_data['time (hours)'], death_data['rep3'], 'o', color='black')
    #plt.xlabel(death_data.columns[0])
    plt.xlabel('Time (hrs)')
    plt.ylabel('Dead cells')
    plt.yscale('log')
    plt.ylim(1e5, 1e7)
    plt.xlim(0, 20)

    plt.tight_layout()
    plt.show()
