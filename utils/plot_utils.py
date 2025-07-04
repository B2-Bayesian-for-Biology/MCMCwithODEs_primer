import matplotlib.pyplot as plt
import arviz as az
from scipy.integrate import odeint
import pymc as pm   
import numpy as np
from scipy.stats import gaussian_kde
import sympy as sp
from tqdm import tqdm


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
    num_prior_samples=500,
    figsize=None,
    hspace=0.5, 
    wspace=0.3
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
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
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
    scaling_factor=2.5,
    fontname='DejaVu Sans',
    fontsize=10,
    figsize=(10, 10),
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

    if figsize is None:
        figsize = (scaling_factor * len(selected_vars), scaling_factor * len(selected_vars))


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
    try:
        plt.tight_layout()
    except Exception:
        pass  # in case it's already tightly controlled
    # Additional spacing if needed
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
    wspace=1,
    figsize=None
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
            # Calculate grid size for square-ish layout
            n_vars = len(selected_vars)
            ncols = int(np.ceil(np.sqrt(n_vars)))
            nrows = int(np.ceil(n_vars / ncols))

            if figsize is None:
                figsize = (4 * ncols, 3 * nrows)

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            axes = np.array(axes).reshape(-1)  # Flatten for easy indexing

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
                    text += f"R={rhat[var].values:.3f} \n "
                if show_ess:
                    text += f"ESS={ess[var].values:.1f} \n "
                if show_geweke and var in geweke:
                    text += f"Geweke z={geweke[var][-1]:.2f}"
                ax.annotate(
                text.strip(),
                xy=(0.5, 0.9),  # was 0.95 — move it lower to avoid overlap with title
                xycoords="axes fraction",
                ha="center",
                va="top",
                fontsize=fontsize - 1,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
                )
                            

            # Hide any unused subplots
            for j in range(i + 1, nrows * ncols):
                fig.delaxes(axes[j])

            # Adjust layout spacing
            plt.subplots_adjust(hspace=hspace, wspace=wspace)

            if save_path:
                plt.savefig(save_path, bbox_inches='tight')

    


    diagnostics = {}
    if show_rhat:
        diagnostics['rhat'] = rhat
    if show_ess:
        diagnostics['ess'] = ess
    if show_geweke:
        diagnostics['geweke'] = geweke

    return diagnostics



### dynamics ###


def posterior_dynamics(
    dataset,
    trace,
    model,
    num_variables,
    num_sigma = None,
    n_plots=100,
    burn_in=0,
    ode_fn=None,
    ode2data_fn=None,
    save_path=None,
    var_properties=None,
    figsize=(5, 5),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None,
    fig_align = 'horizontal',
    color_lines='grey',
    verbose=False
):
    """
    Plot dynamics for each variable with replicates and posterior predictive ODE trajectories.

    Parameters:
        dataset: dict of {variable name: list of dicts with 'time' and 'values'}
        trace: PyMC InferenceData object
        model: PyMC Model object (used to extract free_RVs)
        num_variables: number of ODE state variables
        n_plots: number of posterior samples to use
        burn_in: number of burn-in samples to skip
        ode_fn: function (y, t, params) -> dydt
        ode2data_fn: function (solution) -> dict of derived outputs
        save_path: if provided, path to save figure
        var_properties: dict of customization per variable
        figsize: size per subplot
        show: whether to show the plot
        suptitle: figure-wide title
        verbose: whether to print debug info
    """

    # Safety checks
    if trace is None or model is None:
        raise ValueError("Trace and model must be provided.")
    if dataset is None or not isinstance(dataset, dict):
        raise ValueError("Dataset must be a dict of {var_name: list of replicates}")
    if ode_fn is None or ode2data_fn is None:
        raise ValueError("ODE function and output extractor must be provided.")

    n_chains = trace.posterior.sizes["chain"]
    n_draws_per_chain = trace.posterior.sizes["draw"]
    total_samples = n_chains * n_draws_per_chain

    if burn_in >= total_samples:
        raise ValueError("Burn-in exceeds total number of samples.")
    if n_plots > (total_samples - burn_in):
        raise ValueError("n_plots exceeds available post-burn-in samples.")

    # fixing num_sigma
    if num_sigma is None:
        num_sigma = num_variables

    # Get posterior samples as stacked draws
    posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
    var_names = [v.name for v in model.free_RVs]

    # Assume 1D parameter arrays (exclude multi-dimensional parameters)
    param_matrix = np.vstack([
        posterior_samples[v].values
        for v in var_names
        if v in posterior_samples and posterior_samples[v].values.ndim == 1
    ]).T

    # Plotting setup
    n_vars = len(dataset)
    if fig_align == 'horizontal':
        fig, axes = plt.subplots( 1, n_vars, figsize=(figsize[0]*n_vars, figsize[1]), sharex=sharex, sharey=sharey)
    else:   
        fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1]*n_vars), sharex=sharex, sharey=sharey)
    if n_vars == 1:
        axes = [axes]

    # Get common time range across replicates
    all_times = [t for data in dataset.values() for rep in data for t in rep['time']]
    t_min, t_max = min(all_times), max(all_times)
    time_finer = np.linspace(t_min, t_max, 200)

    for ax, (var_name, replicates) in zip(axes, dataset.items()):
        props = var_properties.get(var_name, {}) if var_properties else {}
        label = props.get("label", var_name)
        color = props.get("color", None)
        ylabel = props.get("ylabel", label)
        xlabel = props.get("xlabel", label)
        sol_key = props.get("sol_key", var_name.lower().replace(" ", "_"))
        is_log = props.get("log", False)
        if is_log:
            ax.set_yscale("log")

        # Posterior predictive ODE simulations
        for i in range(n_plots):
            theta = param_matrix[burn_in + i]

            y0 = theta[-num_variables-num_sigma:-num_sigma]  # assume last 2*num_variables are [y0, bounds]
            ode_params = theta[:-num_variables-num_sigma]  # all but last 2*num_variables

            sol = odeint(ode_fn, y0, t=time_finer, args=(ode_params,), rtol=1e-6, atol=1e-6)
            sol_outputs = ode2data_fn(sol)

            if sol_key not in sol_outputs:
                raise KeyError(f"sol_key '{sol_key}' not found in ode2data_fn output.")

            ax.plot(time_finer, sol_outputs[sol_key], '-', color=color_lines, alpha=0.1)

        # Replicate data
        for i, rep in enumerate(replicates):
            time = rep["time"]
            values = rep["values"]
            rep_label = f"{label} (rep {i+1})" if len(replicates) > 1 else label
            ax.plot(time, values, label=rep_label, color=color, lw=line_width, alpha=alpha, 
                    marker='o', markersize=4, linestyle='None')

        ax.set_title(label, fontsize=fontsize, fontname=fontname)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname=fontname)
        ax.set_xlabel(xlabel, fontsize=fontsize, fontname=fontname)
        ax.tick_params(labelsize=fontsize)
        if len(replicates) > 1:
            ax.legend(fontsize=fontsize - 2)

        

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize+2, fontname=fontname)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes




