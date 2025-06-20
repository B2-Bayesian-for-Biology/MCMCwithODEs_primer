import matplotlib.pyplot as plt
import arviz as az
from scipy.integrate import odeint
import pymc as pm   
import numpy as np
from scipy.stats import gaussian_kde
import sympy as sp

'''
This is preliminary. I will make it more general later.
-- Raunak Dey
'''

########################################
############# Trace ####################
########################################



def plot_trace(
    trace,
    model=None,
    save_path=None,
    var_names_map=None,
    var_order=None,
    fontname='DejaVu Sans',
    fontsize=10,
    prior_color='black',
    num_prior_samples=500
):
    """
    Plot trace with variable renaming, ordering, font options, and prior overlay.

    Parameters:
    - trace: ArviZ InferenceData
    - model: PyMC model (to sample prior automatically)
    - save_path: Path to save the figure
    - var_names_map: dict of {original_name: display_name}
    - var_order: list of variable names in preferred plotting order
    - fontname: Font name for titles
    - fontsize: Font size for titles
    - prior_color: Color for prior histograms
    - num_prior_samples: Number of samples to draw from prior
    """
    # Sample prior from model if provided
    

    if model is not None:
        with model:
            prior_pred = pm.sample_prior_predictive(samples=num_prior_samples)

            # Extract variables from the `prior` group (xarray.Dataset)
            prior_dict = {
                k: prior_pred.prior[k].values
                for k in prior_pred.prior.data_vars
            }

            # Now safely call within the same model context
            prior_idata = pm.to_inference_data(prior=prior_dict)
        
    # Determine plotting variables
    var_names = trace.posterior.data_vars.keys()
    if var_order:
        var_names = [v for v in var_order if v in var_names]
    else:
        var_names = list(var_names)

    labels = [var_names_map.get(v, v) if var_names_map else v for v in var_names]

    # Plot posterior trace
    axes = az.plot_trace(trace, var_names=var_names)#, figsize=(12, len(var_names)*2), show=True)
    chain_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            lines = ax.get_lines()
            for k, line in enumerate(lines):
                line.set_color(chain_colors[k % len(chain_colors)])

            # Set custom title
            if i < len(labels):
                ax.set_title(labels[i], fontname=fontname, fontsize=fontsize)

            # Always show y-axis for trace panels
            if j == 0:
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_tick_params(labelsize=fontsize)

            # Add prior distribution to histogram panel (j == 1)
            if prior_idata and j == 0:
                var = var_names[i]
                if var in prior_idata.prior.data_vars:
                    prior_vals = prior_idata.prior[var].values.flatten()

                    # Add prior KDE to histogram panel
                    kde = gaussian_kde(prior_vals, bw_method=1.0)   # lower = smoother, default is ~0.5–1.0
                    
                    x_min, x_max = prior_vals.min(), prior_vals.max()
                    x_pad = 0.05 * (x_max - x_min)  # add small padding for visual clarity
                    x_vals = np.linspace(x_min - x_pad, x_max + x_pad, 100)
                    y_vals = kde(x_vals)

                    ax.plot(x_vals, y_vals, color=prior_color, alpha=0.7, label='Prior KDE')
                    #ax.fill_between(x_vals, y_vals, color=prior_color, alpha=0.2)
                    '''
                    ax.hist(
                        prior_vals,
                        bins=30,
                        density=True,
                        histtype='stepfilled',
                        alpha=0.3,
                        color=prior_color,
                        label='Prior'
                    )
                    '''    
                    
    
    # Fix layout (less aggressive than tight_layout)
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    if save_path:
        plt.savefig(save_path)
    plt.show()




############################################
# posteriors
############################################
def plot_posterior_pairs(
    trace,
    var_names=None,
    var_names_map=None,
    var_order=None,
    save_path=None,
    fontname='DejaVu Sans',
    fontsize=10,
    figsize=(8, 8),
    plot_kind='kde',  # or 'scatter'
    divergences=True,
    marginals=True,
    hspace=0.4,
    wspace=0.4
):
    """
    Plot posterior pairwise correlations with custom options.

    Parameters:
    - trace: PyMC InferenceData object (posterior samples)
    - var_names: list of variable names to include (optional)
    - var_names_map: dict to rename variables (optional)
    - var_order: list of variable names in desired plotting order (optional)
    - save_path: path to save the plot (optional)
    - fontname: font for labels and titles
    - fontsize: font size for labels and ticks
    - figsize: tuple, figure size
    - plot_kind: 'kde' or 'scatter' for the plot style
    - divergences: show divergent samples (default: True)
    - marginals: show marginal distributions on diagonal (default: True)
    - hspace, wspace: control subplot spacing
    """

    # Get full list of variable names
    available_vars = list(trace.posterior.data_vars)

    # Filter and order variables
    if var_names:
        selected_vars = [v for v in var_names if v in available_vars]
    else:
        selected_vars = available_vars

    if var_order:
        selected_vars = [v for v in var_order if v in selected_vars]

    # Apply variable renaming for plotting
    rename_dict = {v: var_names_map.get(v, v) if var_names_map else v for v in selected_vars}

    # Actually plot
    g = az.plot_pair(
        trace,
        var_names=selected_vars,
        kind=plot_kind,
        divergences=divergences,
        marginals=marginals,
        figsize=figsize
    )

    # Style each subplot
    for ax in g.flatten():
        if ax:
            ax.tick_params(labelsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontname=fontname, fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontname=fontname, fontsize=fontsize)

            # Rename variables in axis labels
            xlabel = ax.get_xlabel()
            ylabel = ax.get_ylabel()
            if xlabel in rename_dict:
                ax.set_xlabel(rename_dict[xlabel], fontname=fontname, fontsize=fontsize)
            if ylabel in rename_dict:
                ax.set_ylabel(rename_dict[ylabel], fontname=fontname, fontsize=fontsize)

    # Adjust layout spacing
    plt.subplots_adjust(hspace=hspace, wspace=wspace)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


############################################
# Convergence
############################################

def plot_convergence(
    trace,
    var_names=None,
    var_order=None,
    var_names_map=None,
    thin=1,
    fontname='DejaVu Sans',
    fontsize=10,
    save_path=None,
    show_geweke=True,
    show_rhat=True,
    show_ess=True,
    show_autocorr=True,
    combine_chains=True,
    max_lag=100,
    hspace=1,
    wspace=1
):
    """
    Plot convergence diagnostics with customization.

    Parameters:
    - trace: ArviZ InferenceData
    - var_names: Variables to include (optional)
    - var_order: Order to plot variables (optional)
    - var_names_map: dict {original_name: display_name}
    - thin: int, show every nth sample in plots
    - fontname, fontsize: Font style for titles/ticks
    - save_path: Path to save the figure
    - show_geweke, show_rhat, show_ess: Include stats annotations
    - show_autocorr: Whether to show autocorrelation plots
    - max_lag: Maximum lag for autocorrelation
    """

    # Filter and order variables
    available_vars = list(trace.posterior.data_vars)
    if var_names:
        selected_vars = [v for v in var_names if v in available_vars]
    else:
        selected_vars = available_vars

    if var_order:
        selected_vars = [v for v in var_order if v in selected_vars]

    labels = [var_names_map.get(v, v) if var_names_map else v for v in selected_vars]

    # Compute diagnostics
    rhat = az.rhat(trace, var_names=selected_vars)
    ess = az.ess(trace, var_names=selected_vars)
    if show_geweke:
        geweke = pm.geweke(trace, var_names=selected_vars)

    # Print diagnostics summary
    print("Convergence Diagnostics:\n")
    for var in selected_vars:
        r = rhat[var].values if show_rhat else None
        e = ess[var].values if show_ess else None
        g = geweke[var] if show_geweke and var in geweke else None
        print(f"{var}: R-hat={r:.3f} | ESS={e:.1f}" + (f" | Geweke z={g[-1]:.2f}" if g is not None else ""))

    if show_autocorr:
        # Plot autocorrelations
        fig, axes = plt.subplots(len(selected_vars), 1, figsize=(8, 2.5 * len(selected_vars)))
        if len(selected_vars) == 1:
            axes = [axes]

        for i, var in enumerate(selected_vars):
            ax = axes[i]
            az.plot_autocorr(trace, var_names=[var], max_lag=max_lag, ax=ax, combined=combine_chains)
            label = labels[i]
            ax.set_title(f"{label}", fontname=fontname, fontsize=fontsize)
            ax.set_xlabel("Lag", fontsize=fontsize)
            ax.set_ylabel("Autocorrelation", fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)

            # Annotate diagnostics
            text = ""
            if show_rhat:
                text += f"R={rhat[var].values:.3f}  "
            if show_ess:
                text += f"ESS={ess[var].values:.1f}  "
            if show_geweke and var in geweke:
                text += f"Geweke z={geweke[var][-1]:.2f}"
            ax.annotate(text.strip(), xy=(0.99, 0.95), xycoords="axes fraction",
                        ha="right", va="top", fontsize=fontsize - 1,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        # Adjust layout spacing
        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        #plt.tight_layout(h_pad=1.2)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

    
    diagnostics = {}
    if show_rhat:
        diagnostics['rhat'] = rhat
    if show_ess:
        diagnostics['ess'] = ess
    if show_geweke:
        diagnostics['geweke'] = geweke

    return diagnostics


def posterior_dynamics():
    raise NotImplementedError("This function is not yet implemented. Please implement the dynamics plotting logic.")



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




'''
def plot_trace2(trace, save_path=None):
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




def plot_posterior(trace, save_path=None):
    
    az.plot_pair(trace, kind='kde', divergences=True, marginals=True)
    if save_path:
        plt.savefig(save_path)
    plt.show()



def plot_autocorrelation(trace, save_path=None):
    rhat = az.rhat(trace)
    autocorr = az.plot_autocorr(trace)
    print(f'Rhat:\n{rhat}\n')
    if save_path:
        plt.savefig(save_path)
    plt.show()

    
'''