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
            # Calculate grid size for square-ish layout
            n_vars = len(selected_vars)
            ncols = int(np.ceil(np.sqrt(n_vars)))
            nrows = int(np.ceil(n_vars / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
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
                    text += f"R={rhat[var].values:.3f}  "
                if show_ess:
                    text += f"ESS={ess[var].values:.1f}  "
                if show_geweke and var in geweke:
                    text += f"Geweke z={geweke[var][-1]:.2f}"
                ax.annotate(text.strip(), xy=(0.99, 0.95), xycoords="axes fraction",
                            ha="right", va="top", fontsize=fontsize - 1,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

            # Hide any unused subplots
            for j in range(i + 1, nrows * ncols):
                fig.delaxes(axes[j])

            # Adjust layout spacing
            plt.subplots_adjust(hspace=hspace, wspace=wspace)

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




def posterior_dynamics2(
    dataset,
    ode_var_names,
    time_vec = None,
    model=None,
    trace=None,
    ode_fn=None,
    num_lines=100,
    time_key="time",
    value_key="values",
    ode_state_count=1,
    save_path=None,
    var_properties=None,
    figsize=(10, 4),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None,
):
    """
    Plot data replicates and posterior ODE dynamics.

    Parameters:
    - dataset: dict of {var: [replicate dicts with 'time' and 'values']}
    - model: PyMC model (to access ODE time vector)
    - trace: ArviZ InferenceData (with posterior samples)
    - ode_fn: function (y, t, params) → dydt
    - num_lines: number of posterior trajectories to simulate
    - ode_var_names: names of variables used in ODE parameters (e.g., ["N0", "mum"])
    - ode_state_count: number of ODE state variables
    - Others: plotting customization
    """

    n_vars = len(dataset)
    fig, axes = plt.subplots(
        n_vars, 1, figsize=(figsize[0], figsize[1]*n_vars),
        sharex=sharex, sharey=sharey
    )
    if n_vars == 1:
        axes = [axes]

    for ax, (var_name, replicates) in zip(axes, dataset.items()):
        props = var_properties.get(var_name, {}) if var_properties else {}
        label = props.get("label", var_name)
        color = props.get("color", None)
        ylabel = props.get("ylabel", label)

        # Plot all replicates
        for i, rep in enumerate(replicates):
            t_data = rep[time_key]
            y_data = rep[value_key]
            rep_label = f"{label} (rep {i+1})" if len(replicates) > 1 else label
            ax.plot(t_data, y_data, label=rep_label, color=color, lw=line_width, alpha=alpha)

        # Plot posterior simulations if model, trace and ODE are given
        if model and trace and ode_fn:
            #time_vec = model.basic_RVs[0].owner.op.times.eval()  # pull time points from DifferentialEquation
            posterior = trace.posterior

            N = posterior.dims["draw"] * posterior.dims["chain"]
            idx = np.random.choice(N, size=num_lines, replace=False)

            # Combine chain and draw
            combined_trace = posterior.stack(sample=("chain", "draw"))

            for s in idx:
                # Extract parameter values
                param_vals = [combined_trace[v].values[s].item() for v in ode_var_names]
                y0 = [param_vals[0]]  # initial condition
                theta = param_vals[1:]  # rest of parameters
                sol = odeint(ode_fn, y0, time_vec, args=(theta,))
                ax.plot(time_vec, sol[:, 0], color=color, alpha=0.2, lw=1, zorder=1)

        ax.set_title(label, fontsize=fontsize, fontname=fontname)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname=fontname)
        ax.tick_params(labelsize=fontsize)

        if len(replicates) > 1:
            ax.legend(fontsize=fontsize-2)

    axes[-1].set_xlabel("Time", fontsize=fontsize, fontname=fontname)

    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize+2, fontname=fontname)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes


def posterior_dynamics(
    dataset,
    trace,
    model,
    num_variables,
    n_plots=100,
    burn_in=0,
    ode_fn=None,
    ode2data_fn=None,
    save_path=None,
    var_properties=None,
    figsize=(10, 4),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import odeint
    import pymc as pm

    # Validate inputs
    n_chains = trace.posterior.sizes["chain"]
    n_draws_per_chain = trace.posterior.sizes["draw"]
    total_samples = n_chains * n_draws_per_chain

    if burn_in >= total_samples:
        raise ValueError("Burn-in exceeds total number of samples in trace.")
    if n_plots > total_samples - burn_in:
        raise ValueError("Number of plots exceeds available samples after burn-in.")
    if not isinstance(dataset, dict):
        raise ValueError("Dataset should be a dictionary of variable replicates.")
    if not isinstance(model, pm.Model):
        raise ValueError("Model should be a PyMC Model object.")

    # Stack trace samples (1d array only, matrix parameter cannot be done like this)
    posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
    posterior_array = np.vstack([
        posterior_samples[v].values
        for v in posterior_samples.data_vars
        if posterior_samples[v].values.ndim == 1
    ]).T

    # Find global time bounds across all replicates
    all_times = []
    for reps in dataset.values():
        for rep in reps:
            all_times.extend(rep["time"])
    min_time = min(all_times)
    max_time = max(all_times)
    time_finer = np.linspace(min_time, max_time, 100)

    # Use one forward pass to get output variable names
    sample_theta = posterior_array[0]
    y0 = sample_theta[-num_variables:]
    sol = odeint(ode_fn, y0, t=time_finer, args=(sample_theta[:-num_variables],), rtol=1e-6, atol=1e-6)
    sol_outputs = ode2data_fn(sol)
    output_varnames = list(sol_outputs.keys())

    # Set up subplots for each output variable
    n_output_vars = len(output_varnames)
    fig, axes = plt.subplots(n_output_vars, 1, figsize=(figsize[0], figsize[1]*n_output_vars),
                             sharex=sharex, sharey=sharey)
    if n_output_vars == 1:
        axes = [axes]

    

    # Overlay replicates from dataset
    for ax, var_name in zip(axes, output_varnames):
        replicates = dataset.get(var_name, [])
        props = var_properties.get(var_name, {}) if var_properties else {}
        label = props.get("label", var_name)
        color = props.get("color", None)
        ylabel = props.get("ylabel", label)

        for i, rep in enumerate(replicates):
            time = rep['time']
            values = rep['values']
            rep_label = f"{label} (rep {i+1})" if len(replicates) > 1 else label
            ax.plot(time, values, label=rep_label, color=color, lw=line_width,
                    alpha=alpha, marker='o', markersize=4, linestyle='None')

        ax.set_title(label, fontsize=fontsize, fontname=fontname)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname=fontname)
        ax.tick_params(labelsize=fontsize)
        if len(replicates) > 1:
            ax.legend(fontsize=fontsize-2)


    # Plot posterior predictive curves
    for i in range(n_plots):
        theta = posterior_array[burn_in + i]
        y0 = theta[-num_variables:]
        sol = odeint(ode_fn, y0, t=time_finer, args=(theta[:-num_variables],), rtol=1e-6, atol=1e-6)
        sol_outputs = ode2data_fn(sol)

        for ax, var_name in zip(axes, output_varnames):
            values = sol_outputs[var_name]
            ax.plot(time_finer, values, '-', color='gray', alpha=0.1)


    axes[-1].set_xlabel("Time", fontsize=fontsize, fontname=fontname)

    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize+2, fontname=fontname)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes

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





def posterior_dynamics3(
    dataset,
    trace,
    model,
    num_variables,
    n_plots=100,
    burn_in=0,
    ode_fn=None,
    ode2data_fn=None,
    save_path=None,
    var_properties=None,
    figsize=(10, 4),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None
):
    """
    Plot dynamics for each variable with its replicates.

    Parameters:
    - dataset: dict of {var: [replicate dicts with 'time' and 'values']}
    - save_path: if set, saves the figure to this path
    - var_properties: dict with optional customizations per variable:
        {
          "Total cells": {"label": "Total", "color": "blue", "ylabel": "Cell count"},
          ...
        }
    - figsize: tuple, size of each subplot (w, h)
    - fontname, fontsize: control font appearance
    - line_width: thickness of replicate lines
    - alpha: transparency of replicate lines
    - show: whether to call plt.show()
    - sharex, sharey: whether to share x or y axes
    - suptitle: optional figure-level title
    """

    # edge cases
    n_chains = trace.posterior.sizes["chain"]
    n_draws_per_chain = trace.posterior.sizes["draw"]
    total_samples = n_chains * n_draws_per_chain

    # error statements
    if burn_in >= total_samples:
        raise ValueError("Burn-in exceeds total number of samples in trace.")
    
    if n_plots > total_samples - burn_in:
        raise ValueError("Number of plots exceeds available samples after burn-in.")

    if dataset is None or len(dataset) == 0:
        raise ValueError("Dataset is empty or not provided.")
    
    if dataset is not None and not isinstance(dataset, dict):
        raise ValueError("Dataset should be a dictionary with variable names as keys.")
    
    if model is None:
        raise ValueError("Model is required to extract posterior samples and dynamics.")
    
    if trace is None:
        raise ValueError("Trace is required to extract posterior samples.")
    
    
    
    if not isinstance(model, pm.Model):
        raise ValueError("Model should be a PyMC Model object.")
    

    n_vars = len(dataset)
    fig, axes = plt.subplots(
        n_vars, 1, figsize=(figsize[0], figsize[1]*n_vars),
        sharex=sharex, sharey=sharey
    )

    if n_vars == 1:
        axes = [axes]  # Ensure axes is iterable

    for ax, (var_name, replicates) in zip(axes, dataset.items()):
        props = var_properties.get(var_name, {}) if var_properties else {}

        label = props.get("label", var_name)
        color = props.get("color", None)
        ylabel = props.get("ylabel", label)


        ### write posterior predictive checks here.
        # Stack chains and draws
        posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
        posterior_array = np.vstack([
            posterior_samples[v].values
            for v in posterior_samples.data_vars
            if posterior_samples[v].values.ndim == 1
        ]).T
        

        all_times = []

        for variable_data in dataset.values():
            for replicate in variable_data:
                all_times.extend(replicate["time"])

        min_time = min(all_times)
        max_time = max(all_times)

        for i in range(n_plots):
            theta = posterior_array[i]
            # last two are initial conditions
            y0 = theta[-num_variables:] # initial live and dead cells
            time_finer = np.linspace(min_time, max_time, 100)
            sol = odeint(ode_fn, y0, t=time_finer, args=(theta[:-num_variables],), rtol=1e-6, atol=1e-6)
            sol_outputs = ode2data_fn(sol)
            # Extract live and dead cells from the solution

        #fig, axes = plt.subplots(1, num_variables, figsize=(5 * n_vars, 4), squeeze=False)

        #for i, (var_name, values) in enumerate(sol_outputs.items()):
        #    ax = axes[0, i]
        #    ax.plot(time_finer, values, '-', color='gray', alpha=0.1)
        #    ax.set_title(var_name)
        #    ax.set_xlabel("Time")
        #    ax.set_ylabel("Value")
        

        # Plot all replicates data
        for i, rep in enumerate(replicates):
            time = rep['time']
            values = rep['values']
            rep_label = f"{label} (rep {i+1})" if len(replicates) > 1 else label
            ax.plot(time, values, label=rep_label, color=color, lw=line_width, alpha=alpha, 
                    marker='o', markersize=4, linestyle='None')

        ax.set_title(label, fontsize=fontsize, fontname=fontname)
        ax.set_ylabel(ylabel, fontsize=fontsize, fontname=fontname)
        ax.tick_params(labelsize=fontsize)

        if len(replicates) > 1:
            ax.legend(fontsize=fontsize-2)

    axes[-1].set_xlabel("Time", fontsize=fontsize, fontname=fontname)

    if suptitle:
        plt.suptitle(suptitle, fontsize=fontsize+2, fontname=fontname)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes






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