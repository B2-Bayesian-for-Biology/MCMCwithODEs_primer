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



def plot_trace2(
    trace,
    save_path=None,
    var_names_map=None,
    var_order=None,
    fontname='DejaVu Sans',
    fontsize=10,
    prior_trace=None,
    prior_color='gray'
):
    """
    Enhanced trace plot with customization options.

    Parameters:
    - trace: ArviZ InferenceData object
    - save_path: Path to save the plot (optional)
    - var_names_map: dict mapping original variable names to new labels
    - var_order: list specifying the order of variables to plot
    - fontname: font name for labels
    - fontsize: font size for labels
    - prior_trace: ArviZ InferenceData object with prior samples
    - prior_color: color for prior distribution overlay
    """
    # Determine the variables to plot
    var_names = trace.posterior.data_vars.keys()
    if var_order:
        var_names = [v for v in var_order if v in var_names]
    else:
        var_names = list(var_names)

    # Apply custom variable names if given
    labels = [var_names_map.get(v, v) if var_names_map else v for v in var_names]

    # Plot trace
    axes = az.plot_trace(trace, var_names=var_names, figsize=(12, len(var_names)*2), show=True)

    chain_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            lines = ax.get_lines()
            for k, line in enumerate(lines):
                line.set_color(chain_colors[k % len(chain_colors)])
            
            # Set custom font and label
            if j == 0 and i < len(labels):
                ax.set_title(labels[i], fontname=fontname, fontsize=fontsize)

            # Overlay prior, if available
            if prior_trace and j == 1:  # Histogram plot
                var_name = var_names[i]
                if var_name in prior_trace.prior.data_vars:
                    prior_vals = prior_trace.prior[var_name].values.flatten()
                    ax.hist(
                        prior_vals,
                        bins=30,
                        density=True,
                        histtype='stepfilled',
                        alpha=0.3,
                        color=prior_color,
                        label='Prior'
                    )
                    ax.legend(fontsize=fontsize-2)

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

