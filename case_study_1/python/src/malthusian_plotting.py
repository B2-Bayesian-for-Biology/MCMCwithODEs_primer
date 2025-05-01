from malthusian import *

# ---------------------------
# PLOTTING FUNCTIONS
# ---------------------------

# Plot the original data
def plot_data(ax, time, obs, **kwargs):
    ax.plot(time, obs, 'ko', label='data', **kwargs)

# Plot a single model realization
def plot_model(ax, model_output, time, **kwargs):
    ax.plot(time, model_output, **kwargs)

# Overlay posterior predictive simulations
def plot_posterior_predictive(ax, trace, time, obs, num_samples=200, ode_solver=solve_cells_ode, **kwargs):
    df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, time, obs, lw=0)
    for _, row in df.iterrows():
        model_output = ode_solver(row["N0"], time, row["mum"])
        plot_model(ax, model_output, time, lw=1, alpha=0.1, c='b', **kwargs)
    ax.legend()
    ax.semilogy()

# ---------------------------
# MAIN EXECUTION
# ---------------------------

# Load data
data = pd.read_csv("./../data/phaeocystis_control.csv")
time = data['times'].values
obs = data['cells'].values
trace = az.from_netcdf('../data/posterior_trace.nc')

# create figure
fig = plt.figure(figsize=(12, 15))
outer_gs = fig.add_gridspec(3, 1, height_ratios=[4, 3,3])  # Top 1/2 and bottom 1/2

# --- Top Half: Two Equal Columns ---
gs_top = outer_gs[0].subgridspec(1, 2)
ax_model = fig.add_subplot(gs_top[0, 0])
ax_walks = fig.add_subplot(gs_top[0, 1])

# --- Bottom Half: Two Rows, Each Split 2/3 and 1/3 ---
gs_bottom = outer_gs[1:3].subgridspec(2, 3, width_ratios=[1, 1, 1])

ax_trace1 = fig.add_subplot(gs_bottom[0, 0:2])
ax_hist1 = fig.add_subplot(gs_bottom[0, 2])

ax_trace2 = fig.add_subplot(gs_bottom[1, 0:2])
ax_hist2 = fig.add_subplot(gs_bottom[1, 2])

# Random walk scatter
mum = trace.posterior["mum"].values.flatten()
N0 = trace.posterior["N0"].values.flatten()
ax_walks.scatter(mum, N0, s=1, alpha=0.5)

# trace plots and KDE plots
az.plot_trace(trace, var_names=["mum"], axes=np.r_[[[ax_hist1],[ax_trace1]]].T, compact=True, show=False)
az.plot_posterior(trace, var_names=["mum"], ax=ax_hist1, show=False)
az.plot_trace(trace, var_names=["N0"], axes=np.r_[[[ax_hist2],[ax_trace2]]].T, compact=True, show=False)
az.plot_posterior(trace, var_names=["N0"], ax=ax_hist2, show=False)
plot_posterior_predictive(ax_model, trace, time, obs)

plt.tight_layout()

fig.savefig('../figures/figure1')
