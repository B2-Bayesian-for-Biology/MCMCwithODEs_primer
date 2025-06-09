from malthusian_death_constrained import *
import matplotlib.pyplot as plt

death_trace = az.from_netcdf('../data/death_posterior_trace_constrained.nc')

############################################
# trace plots
############################################

# plot chains - DEAD
axes = az.plot_trace(death_trace)
chain_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for ax_row in axes:
    for ax in ax_row:
        lines = ax.get_lines()
        for i, line in enumerate(lines):
            line.set_color(chain_colors[i % len(chain_colors)])
plt.tight_layout()
plt.savefig('../figures/death_chains_constrained')

############################################
# posteriors
############################################

# Plot posterior correlations
az.plot_pair(death_trace, kind='kde', divergences=True, marginals=True)
plt.savefig('../figures/death_posterior_constrained')

############################################
# autocorrelation
############################################

# 7 Convergence test for the chains - Gelman-Rubin, Geweke, and Autocorrelation
rhat = az.rhat(death_trace)
autocorr = az.plot_autocorr(death_trace)
print(f'Rhat:\n{rhat}\n')
plt.savefig('../figures/death_autocorrelation_constrained')

############################################
# dynamics
############################################

# Plot the original data
def plot_data(ax, time, obs, **kwargs):
    ax.plot(time, obs, 'ko', label=r'$Phaeocystis$ $globosa$',zorder=1, **kwargs)

# Plot a single model realization
def plot_model(ax, model_output, time, **kwargs):
    ax.plot(time, model_output, **kwargs)

# Overlay posterior predictive simulations
def plot_posterior_predictive(ax, trace, time, obs, num_samples=400, ode_solver=cells_ode, **kwargs):
    df = az.extract(trace, num_samples=num_samples).to_dataframe()
    plot_data(ax, time, obs, lw=0)
    for _, row in df.iterrows():
        model_output = odeint(ode_solver,[row["N0"]], time, args=([row["mum"],row["delta"]],))
        plot_model(ax, model_output, time, lw=1, alpha=0.1, c='g',zorder=0, **kwargs)
    ax.legend()
    ax.semilogy()

f,ax = plt.subplots()
data = pd.read_csv("./../data/phaeocystis_control.csv")
time = data['times'].values
obs = data['cells'].values
plot_posterior_predictive(ax, death_trace, time, obs)

f.savefig('../figures/death_dynamics_constrained')


