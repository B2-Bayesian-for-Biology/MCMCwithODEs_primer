from generic_model import *

# ---------------------------
# PLOTTING
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

# This block only runs if the script is executed directly (not when imported)
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("./../data/phaeocystis_control.csv")
    time = data['times'].values
    obs = data['cells'].values

    trace = az.from_netcdf('../data/posterior_trace.nc')

    # Plot results
    fig, ax = plt.subplots()
    plot_posterior_predictive(ax, trace, time, obs)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cell density")
    ax.set_title("Posterior Predictive Fits")
    fig.savefig('../figures/figure1')
    plt.show()


