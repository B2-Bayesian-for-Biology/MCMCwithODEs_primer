import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

from pymc.ode import DifferentialEquation

# =========================================================
# Load data
# =========================================================

dataset = pd.read_csv("./../data/total_cells.csv")

ehux_cells = dataset.tail(15)

ehux_total_time    = ehux_cells['Time (days)'].values
ehux_total_density = 1e6 * ehux_cells[' Density (1e6/ml)'].values

death_dataset = pd.read_csv("./../data/death_percentage.csv")
ehux_death = death_dataset.tail(15)

ehux_dead_time    = ehux_death['Time (days)'].values
ehux_dead_density = ehux_death[' Dead percentage '].values * ehux_total_density / 100


# =========================================================
# ODE system (SYMBOLIC SAFE)
# =========================================================

def monod_ode(y, t, theta):
    # --- Explicit indexing (REQUIRED) ---
    N = y[0]
    P = y[1]
    D = y[2]

    mu_max = theta[0]
    Ks     = theta[1]
    Qn     = theta[2]
    delta  = theta[3]

    mu = mu_max * N / (N + Ks)

    dNdt = -Qn * mu * P
    dPdt = mu * P - delta * P
    dDdt = delta * P

    return [dNdt, dPdt, dDdt]


# =========================================================
# Differentiable ODE solver
# =========================================================

ode_model = DifferentialEquation(
    func=monod_ode,
    times=ehux_total_time,
    n_states=3,
    n_theta=4,
    t0=ehux_total_time[0],
)


# =========================================================
# PyMC model (NUTS)
# =========================================================

with pm.Model() as model:

    # ---- Parameters ----
    mu_max = pm.Uniform("mu_max", 0.4, 0.7)
    Ks     = pm.Uniform("Ks", 0.05, 0.4)
    Qn     = pm.Uniform("Qn", 2e-10, 30e-10)
    delta  = pm.Normal("delta", mu=0.034, sigma=0.003)

    # ---- Initial conditions ----
    N0 = pm.TruncatedNormal("N0", mu=880, sigma=100, lower=200, upper=2000)
    P0 = pm.Normal("P0", mu=211_017, sigma=14_380)
    D0 = pm.Normal("D0", mu=31_104,  sigma=5_761)

    y0 = pt.stack([N0, P0, D0])
    theta = pt.stack([mu_max, Ks, Qn, delta])

    # ---- Solve ODE ----
    sol = ode_model(y0=y0, theta=theta)

    N = sol[:, 0]
    P = sol[:, 1]
    D = sol[:, 2]

    total = P + D
    dead  = D

    # ---- Likelihood ----
    eps = 1e-12

    sigma_live = pm.HalfNormal("sigma_live", 1)
    sigma_dead = pm.HalfNormal("sigma_dead", 1)

    pm.Normal(
        "Y_total",
        mu=pt.log(total + eps),
        sigma=sigma_live,
        observed=np.log(ehux_total_density),
    )

    pm.Normal(
        "Y_dead",
        mu=pt.log(dead + eps),
        sigma=sigma_dead,
        observed=np.log(ehux_dead_density),
    )

    # ---- Sample with NUTS ----
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        cores=4,
        return_inferencedata=True,
    )


# =========================================================
# Diagnostics
# =========================================================

print(az.summary(trace, round_to=3))
az.plot_trace(trace)
plt.show()