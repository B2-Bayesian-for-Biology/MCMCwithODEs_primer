import matplotlib.pyplot as plt
import arviz as az
from scipy.integrate import odeint
import pymc as pm   
import numpy as np

'''
This is preliminary. I will make it more general later.
-- Raunak Dey
'''

def plot_trace(trace, save_path=None):
    """
    Plot the trace of a PyMC trace object.

    Parameters:
    - trace: PyMC trace object
    - save_path: Path to save the plot (optional)
    """
    axes = az.plot_trace(trace)
    chain_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for ax_row in axes:
        for ax in ax_row:
            lines = ax.get_lines()
            for i, line in enumerate(lines):
                line.set_color(chain_colors[i % len(chain_colors)])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()




############################################
# posteriors
############################################

def plot_posterior(trace, save_path=None):
    """
    Plot posterior correlations of a PyMC trace object.

    Parameters:
    - trace: PyMC trace object
    - save_path: Path to save the plot (optional)
    """
    az.plot_pair(trace, kind='kde', divergences=True, marginals=True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


############################################
# autocorrelation
############################################

def plot_autocorrelation(trace, save_path=None):
    """
    Plot the autocorrelation of a PyMC trace object.

    Parameters:
    - trace: PyMC trace object
    - save_path: Path to save the plot (optional)
    """
    rhat = az.rhat(trace)
    autocorr = az.plot_autocorr(trace)
    print(f'Rhat:\n{rhat}\n')
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_dynamics(model, trace, y_vector, time, n_plot, save_path=None):

    posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
    posterior_array = np.vstack([
    posterior_samples["$r$ (growth rate)"].values,
    posterior_samples["$K$ (carrying capacity)"].values,
    posterior_samples["$\delta$ (death rate)"].values,
    posterior_samples["$P_0$ (init. live)"].values,
    posterior_samples["$D_0$ (init. dead)"].values
    ]).T  # Shape: (n_samples, 5)




def run_posterior_predictive_checks(model, trace, var_names=None, plot=True, savepath=None):
    """
    Run posterior predictive checks on a PyMC model.

    Parameters:
    - model: a PyMC model
    - trace: inference data from pm.sample (with return_inferencedata=True)
    - var_names: list of observed variable names to sample (e.g., ["Y_live", "Y_dead"])
    - plot: whether to plot the posterior predictive checks
    - savepath: if provided, saves the plot to the given file path
    """
    with model:
        ppc = pm.sample_posterior_predictive(trace, var_names=var_names)

    if plot and var_names is not None:
        for var in var_names:
            obs_data = trace.observed_data[var].values
            pred_samples = ppc.posterior_predictive[var].values  # <-- fixed here

            plt.figure(figsize=(8, 5))
            plt.plot(obs_data, 'o-', label="Observed", color="black")
            n_samples = min(50, len(pred_samples))
            for s in pred_samples[np.random.choice(len(pred_samples), size=n_samples, replace=False)]:
                plt.plot(s, alpha=0.1, color="blue")
            plt.title(f"Posterior Predictive Check: {var}")
            plt.xlabel("Time Index")
            plt.ylabel("log(cell count)")
            plt.legend()
            plt.grid(True)

            if savepath:
                plt.savefig(f"{savepath}_{var}.png")
            else:
                plt.show()

    return ppc

