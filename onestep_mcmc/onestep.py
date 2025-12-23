# ===============================
# Imports
# ===============================
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
from scipy.integrate import solve_ivp
import arviz as az
import os
import matplotlib.pyplot as plt

# ===============================
# Data
# ===============================
free_phages = np.array([
    [13200, 13500, 15400],
    [15500, 11800, 14450],
    [17200, 13100, 15000],
    [12100, 13000, 14500],
    [16700, 12900, 13500],
    [194000, 60000, 67000],
    [2860000, 1280000, 1410000],
    [12200000, 11100000, 10100000],
    [42000000, 27300000, 25350000],
    [49000000, 53000000, 47000000],
    [58500000, 50500000, 61000000],
    [42500000, 33000000, 44000000],
])

time = np.array([
    0.0, 0.33333333, 0.66666667, 1.0, 1.16666667, 1.33333333,
    1.5, 1.66666667, 1.83333333, 2.0, 2.33333333, 2.66666667
])

phage_obs = np.mean(free_phages, axis=1)

# ===============================
# Constants
# ===============================
NE_FIXED = 173
S0 = 1e8
V0 = 1e7
r_fixed = 0.3
dil_time = 0.25  # 15 minutes
dil_factor = 100

# ===============================
# One-step phage ODE
# ===============================
def phage_onestep_ivp(t, y, params):
    phi, beta, tau, r = params
    NE = len(y) - 3

    S = y[0]
    E = y[1:NE+1]
    I = y[NE+1]
    V = y[NE+2]

    etaeff = (NE + 1) / tau

    dSdt = r * S - phi * V * S

    dEdt = np.zeros_like(E)
    dEdt[0] = phi * S * V - etaeff * E[0]
    if NE > 1:
        dEdt[1:] = etaeff * E[:-1] - etaeff * E[1:]

    dIdt = etaeff * (E[-1] - I)
    dVdt = beta * etaeff * I - phi * V * (S + I + np.sum(E))

    dydt = np.zeros_like(y)
    dydt[0] = dSdt
    dydt[1:NE+1] = dEdt
    dydt[NE+1] = dIdt
    dydt[NE+2] = dVdt

    return dydt

# ===============================
# SolveIVP Wrapper (2-stage protocol)
# ===============================
class SolveIVPWrapper(Op):
    itypes = [pt.dvector]
    otypes = [pt.dmatrix]

    def __init__(self, times, NE):
        self.times = times
        self.NE = NE

    def perform(self, node, inputs, outputs):
        theta_y0, = inputs

        phi, beta, tau, r = theta_y0[:4]
        y0 = theta_y0[4:]

        # ---- Stage 1: 15 min interaction ----
        sol_pre = solve_ivp(
            fun=lambda t, y: phage_onestep_ivp(
                t, y, [phi, beta, tau, r]
            ),
            t_span=(0.0, dil_time),
            y0=y0,
            t_eval=[dil_time],
            method="LSODA"
        )

        if not sol_pre.success:
            outputs[0][0] = np.full((len(self.times), len(y0)), np.nan)
            return

        # ---- Stage 2: 100x dilution ----
        y0_dil = sol_pre.y[:, -1] / dil_factor

        # ---- Stage 3: main experiment ----
        sol = solve_ivp(
            fun=lambda t, y: phage_onestep_ivp(
                t, y, [phi, beta, tau, r]
            ),
            t_span=(self.times[0], self.times[-1]),
            y0=y0_dil,
            t_eval=self.times,
            method="LSODA"
        )

        if not sol.success:
            outputs[0][0] = np.full((len(self.times), len(y0)), np.nan)
            return

        outputs[0][0] = sol.y.T

# ===============================
# PyMC Model
# ===============================
def build_phage_model(times, phage_obs):

    ode_op = SolveIVPWrapper(times, NE_FIXED)

    with pm.Model() as model:

        # ---- Priors ----
        phi = pm.LogNormal("phi", mu=np.log(6.8e-08), sigma=0.5)
        beta = pm.TruncatedNormal("beta", mu=350, sigma=100,lower=1,upper=1000)
        tau = pm.TruncatedNormal("tau", mu=2, sigma=1, lower=0.1, upper=4)

        sigma = pm.HalfNormal("sigma", 0.3)

        # ---- Initial state ----
        y0 = np.zeros(NE_FIXED + 3)
        y0[0] = S0
        y0[-1] = V0

        theta_y0 = pt.concatenate([
            pt.stack([phi, beta, tau, r_fixed]),
            pt.as_tensor_variable(y0)
        ])

        sol = ode_op(theta_y0)
        V_model = sol[:, -1]

        eps = 1e-12
        pm.Normal(
            "Y",
            mu=pt.log10(V_model + eps),
            sigma=sigma,
            observed=np.log10(phage_obs)
        )

    return model

# ===============================
# Run inference
# ===============================
model = build_phage_model(time, phage_obs)


flag_run = True  # Set to False to skip sampling if trace file exists

# Check if the trace file exists
trace_file = "./phage_onestep_trace.nc"
if os.path.exists(trace_file) and flag_run == False:
    print(f"{trace_file} exists. Skip run")
    trace = az.from_netcdf("phage_onestep_trace.nc")
else:
    print(f"{trace_file} does not exist.")
    with model:
        step = pm.Metropolis()
        trace = pm.sample(
            draws=3000,
            tune=2000,
            chains=4,
            step=step,
            random_seed=42,
            return_inferencedata=True
        )
    # Save the trace to a NetCDF file
    trace.to_netcdf("phage_onestep_trace.nc")



# ===============================
# Diagnostics
# ===============================
az.summary(trace, var_names=["phi", "beta", "tau", "sigma"])

# ===============================
# Posterior predictive trajectories
# ===============================
posterior = trace.posterior

'''
az.plot_trace(
    trace,
    var_names=["phi", "beta", "tau"],
    combined=False,
    chain_prop={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]}
)
'''

axes = az.plot_trace(
    trace,
    var_names=["phi", "beta", "tau"],
    chain_prop={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]}
)
plt.savefig("trace_combined.png", dpi=300, bbox_inches="tight")



fig = az.plot_pair(
    trace,
    var_names=["phi", "beta", "tau"],
    kind="kde",
)
plt.savefig("pairwise_kde.png", dpi=300, bbox_inches="tight")


rhat = az.rhat(trace, var_names=["phi", "beta", "tau"])
ess = az.ess(trace, var_names=["phi", "beta", "tau"], method="bulk")

axes = az.plot_autocorr(
    trace,
    var_names=["phi", "beta", "tau"],
    combined=False,
    max_lag=100,
)

for ax_row, var in zip(axes, ["phi", "beta", "tau"]):
    rhat_val = float(rhat[var])
    ess_val = float(ess[var])
    for ax in ax_row:
        ax.set_title(f"{var} | R̂={rhat_val:.3f}, ESS={ess_val:.0f}")

plt.savefig("autocorr_rhat_ess.png", dpi=300, bbox_inches="tight")

# ===============================
plt.figure(figsize=(6,4))
for i in range(200):
    phi_i = posterior["phi"].values[0, i]
    beta_i = posterior["beta"].values[0, i]
    tau_i = posterior["tau"].values[0, i]

    y0 = np.zeros(NE_FIXED + 3)
    y0[0] = S0
    y0[-1] = V0

    sol_pre = solve_ivp(
        lambda t, y: phage_onestep_ivp(t, y, [phi_i, beta_i, tau_i, r_fixed]),
        (0.0, dil_time),
        y0,
        t_eval=[dil_time],
        method="LSODA"
    )

    y0_dil = sol_pre.y[:, -1] / dil_factor

    sol = solve_ivp(
        lambda t, y: phage_onestep_ivp(t, y, [phi_i, beta_i, tau_i, r_fixed]),
        (time[0], time[-1]),
        y0_dil,
        t_eval=time,
        method="LSODA"
    )

    plt.plot(time, sol.y[-1], color="red", alpha=0.05)

plt.scatter(time, phage_obs, color="black", label="Data")
plt.yscale("log")
plt.xlabel("Time (hours)")
plt.ylabel("Free phages")
plt.legend()
plt.savefig("posterior_predictive.png", dpi=300, bbox_inches="tight")
plt.show()