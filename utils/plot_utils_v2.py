import matplotlib.pyplot as plt
import arviz as az
from scipy.integrate import odeint
import pymc as pm   
import numpy as np
from scipy.stats import gaussian_kde
import sympy as sp
from tqdm import tqdm
from scipy.integrate import solve_ivp
from matplotlib.ticker import AutoLocator, AutoMinorLocator, ScalarFormatter

'''
This is preliminary. I will make it more general later.
-- Raunak Dey
'''

def _remove_rug(ax):
    # delete any 'rug' tick marks (Line2D with marker '|')
    for line in list(ax.lines):
        try:
            if line.get_marker() == '|':
                line.remove()
        except Exception:
            pass

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
    Plot trace with variable renaming, ordering, font options, prior overlay,
    and scientific-notation ticks (×10^n) on BOTH axes for ALL panels.
    """

    # --- Helper: apply scientific notation formatter to both axes (new instance per axis) ---
    def apply_sci_mathtext(ax):
        fmt_x = ScalarFormatter(useMathText=True)
        fmt_x.set_powerlimits((-3, 4))  # force ×10^n when appropriate
        ax.xaxis.set_major_formatter(fmt_x)

        fmt_y = ScalarFormatter(useMathText=True)
        fmt_y.set_powerlimits((-3, 4))
        ax.yaxis.set_major_formatter(fmt_y)

        ax.tick_params(axis="x", labelsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

    # --- Sample prior if model is provided ---
    prior_idata = None
    if model is not None:
        with model:
            prior_pred = pm.sample_prior_predictive(samples=num_prior_samples)
            prior_dict = {k: prior_pred.prior[k].values
                          for k in prior_pred.prior.data_vars}
            prior_idata = pm.to_inference_data(prior=prior_dict)

    # --- Determine variables & labels ---
    var_names = list(trace.posterior.data_vars.keys())
    if var_order:
        var_names = [v for v in var_order if v in var_names]
    labels = [var_names_map.get(v, v) if var_names_map else v for v in var_names]

    # --- Plot with ArviZ ---
    axes = az.plot_trace(
        trace,
        var_names=var_names,
        figsize=figsize,
        #plot_kwargs={"rug": False}
    )

    chain_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            # Color per-chain lines consistently
            for k, line in enumerate(ax.get_lines()):
                line.set_color(chain_colors[k % len(chain_colors)])

            # Titles
            if i < len(labels):
                ax.set_title(labels[i], fontname=fontname, fontsize=fontsize)

            # Ensure ticks visible on left panels and label y as Density
            if j == 0:
                _remove_rug(ax)
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_major_locator(AutoLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.set_ylabel("Density", fontsize=fontsize, fontname=fontname)

                # Add prior KDE if available
                if (prior_idata is not None) and (var_names[i] in prior_idata.prior.data_vars):
                    '''
                    prior_vals = prior_idata.prior[var_names[i]].values.flatten()
                    if prior_vals.size > 1:
                        kde = gaussian_kde(prior_vals, bw_method=1.0)
                        x_min, x_max = prior_vals.min(), prior_vals.max()
                        x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
                        x_vals = np.linspace(x_min - x_pad, x_max + x_pad, 200)
                        y_vals = kde(x_vals)
                        ax.plot(x_vals, y_vals, color=prior_color, alpha=0.8, label='Prior KDE')
                    '''
                    # Replace your KDE block with this
                    prior_vals = prior_idata.prior[var_names[i]].values.flatten()
                    if prior_vals.size > 1:
                        a = np.min(prior_vals)
                        b = np.max(prior_vals)
                        if np.isfinite(a) and np.isfinite(b) and b > a:
                            # Draw a flat line for the uniform PDF
                            height = 1.0 / (b - a)
                            ax.hlines(height, a, b, color=prior_color, alpha=0.9, label='Prior (Uniform)')
                            # (optional) mark vertical edges so it reads clearly
                            ax.vlines([a, b], 0, height, color=prior_color, alpha=0.3, linestyle='--')
                        else:
                            # Fallback to KDE if degenerate
                            kde = gaussian_kde(prior_vals, bw_method=1.0)
                            x_vals = np.linspace(a, b, 200)
                            y_vals = kde(x_vals)
                            ax.plot(x_vals, y_vals, color=prior_color, alpha=0.8, label='Prior KDE')
                    
            # Apply scientific notation (×10^n) to BOTH axes on every panel
            apply_sci_mathtext(ax)

    # Layout & save
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    return axes

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


    # Apply thinning and safety check
    posterior = trace.posterior
    n_draws = posterior.sizes["draw"]

    if thin > 1:
        if thin * max_lag > n_draws:
            raise ValueError(
                f"Thinning too aggressive: thin={thin}, max_lag={max_lag}, but only {n_draws} samples available.\n"
                "Reduce 'thin' or 'max_lag'."
            )
        trace = trace.sel(draw=slice(None, None, thin))

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








def posterior_dynamics_solve_ivp(
    dataset,
    trace,
    model,
    num_variables,
    ode_fn,
    ode2data_fn,
    n_plots=100,
    burn_in=0,
    time_resolution=200,
    var_properties=None,
    save_path=None,
    figsize=(5, 5),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    color_lines='grey',
    fig_align='horizontal',
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None,
    verbose=False,
    num_sigma=None
):
    """
    Posterior predictive plotting using solve_ivp.

    Parameters:
        dataset: dict of {var_name: list of dicts with keys "time", "values"}
        trace: PyMC InferenceData object
        model: PyMC Model object (used to extract free_RVs)
        num_variables: number of state variables in the ODE system
        ode_fn: function (t, y, theta) -> dy/dt
        ode2data_fn: function (sol) -> dict of named outputs
    """
    import arviz as az

       # fixing num_sigma
    if num_sigma is None:
        num_sigma = num_variables

    posterior_samples = trace.posterior.stack(draws=("chain", "draw"))
    var_names = [v.name for v in model.free_RVs]

    param_matrix = np.vstack([
        posterior_samples[v].values
        for v in var_names
        if posterior_samples[v].values.ndim == 1
    ]).T

    # Plot layout
    n_vars = len(dataset)
    if fig_align == 'horizontal':
        fig, axes = plt.subplots(1, n_vars, figsize=(figsize[0]*n_vars, figsize[1]), sharex=sharex, sharey=sharey)
    else:
        fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1]*n_vars), sharex=sharex, sharey=sharey)
    if n_vars == 1:
        axes = [axes]

    # Shared time range
    all_times = [t for data in dataset.values() for rep in data for t in rep['time']]
    t_min, t_max = min(all_times), max(all_times)
    time_finer = np.linspace(t_min, t_max, time_resolution)

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

        # Posterior ODE simulations
        for i in range(n_plots):
            theta = param_matrix[burn_in + i]
            
            y0 = theta[-num_variables-num_sigma:-num_sigma]  # assume last 2*num_variables are [y0, bounds]
            ode_params = theta[:-num_variables-num_sigma]  # all but last 2*num_variables
            print(theta)
            #y0 = theta[-num_variables:]
            #ode_params = theta[:-num_variables]

            try:
                sol_ivp = solve_ivp(
                    fun=lambda t, y: ode_fn(t, y, ode_params),
                    t_span=(time_finer[0], time_finer[-1]),
                    y0=y0,
                    t_eval=time_finer,
                    rtol=1e-6,
                    atol=1e-6
                )
                if not sol_ivp.success:
                    if verbose:
                        print(f"[Sample {i}] ODE failed: {sol_ivp.message}")
                    continue

                sol_outputs = ode2data_fn(sol_ivp.y.T)

                if sol_key not in sol_outputs:
                    raise KeyError(f"sol_key '{sol_key}' not found in ode2data_fn output.")

                ax.plot(time_finer, sol_outputs[sol_key], '-', color=color_lines, alpha=0.1)

            except Exception as e:
                if verbose:
                    print(f"[Sample {i}] Exception in solve_ivp: {e}")
                continue

        # Plot replicates
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






import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def posterior_dynamics_solve_ivp_flexible(
    dataset,
    trace,
    y0_dict,
    theta_dict,
    ode_fn,
    ode2data_fn,
    n_plots=100,
    burn_in=0,
    time_resolution=200,
    var_properties=None,
    save_path=None,
    figsize=(5, 5),
    fontname='DejaVu Sans',
    fontsize=12,
    line_width=2,
    alpha=0.8,
    color_lines='grey',
    fig_align='horizontal',
    show=True,
    sharex=False,
    sharey=False,
    suptitle=None,
    verbose=False
):
    """
    Posterior predictive plotting using solve_ivp with flexible y0/theta dicts.

    Parameters:
        y0_dict: dict of initial conditions. Each value is scalar or array (chain, draw)
        theta_dict: dict of parameters. Each value is scalar or array (chain, draw)
    """
    import arviz as az
    import xarray as xr

    # Stack posterior samples
    posterior = trace.posterior.stack(draws=("chain", "draw"))
    total_samples = posterior.draws.size

    if n_plots + burn_in > total_samples:
        raise ValueError("n_plots + burn_in exceeds number of posterior samples")

    # Get common time range
    all_times = [t for v in dataset.values() for rep in v for t in rep["time"]]
    t_min, t_max = min(all_times), max(all_times)
    t_eval = np.linspace(t_min, t_max, time_resolution)

    # Setup figure
    n_vars = len(dataset)
    if fig_align == 'horizontal':
        fig, axes = plt.subplots(1, n_vars, figsize=(figsize[0]*n_vars, figsize[1]), sharex=sharex, sharey=sharey)
    else:
        fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1]*n_vars), sharex=sharex, sharey=sharey)
    if n_vars == 1:
        axes = [axes]

    # Posterior predictive sampling loop
    for i in range(n_plots):
        idx = burn_in + i

        # Extract parameter values from theta_dict
        theta_vals = []
        for key, val in theta_dict.items():
            if isinstance(val, xr.DataArray):
                theta_vals.append(posterior[key].values.flatten()[idx])
            else:
                theta_vals.append(val)

        # Extract initial conditions from y0_dict
        y0_vals = []
        for key, val in y0_dict.items():
            if isinstance(val, xr.DataArray):
                y0_vals.append(posterior[key].values.flatten()[idx])
            else:
                y0_vals.append(val)

        # Solve the ODE
        try:
            sol_ivp = solve_ivp(
                fun=lambda t, y: ode_fn(t, y, theta_vals),
                t_span=(t_eval[0], t_eval[-1]),
                y0=y0_vals,
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-6
            )
            if not sol_ivp.success:
                if verbose:
                    print(f"[Sample {i}] ODE solve failed: {sol_ivp.message}")
                continue

            sol_outputs = ode2data_fn(sol_ivp.y.T)

        except Exception as e:
            if verbose:
                print(f"[Sample {i}] Exception in solve_ivp: {e}")
            continue

        # Plot results
        for ax, (var_name, replicates) in zip(axes, dataset.items()):
            props = var_properties.get(var_name, {}) if var_properties else {}
            label = props.get("label", var_name)
            color = props.get("color", None)
            sol_key = props.get("sol_key", var_name.lower().replace(" ", "_"))
            is_log = props.get("log", False)

            if sol_key not in sol_outputs:
                raise KeyError(f"{sol_key} not in ode2data_fn output")

            if is_log:
                ax.set_yscale("log")

            ax.plot(t_eval, sol_outputs[sol_key], '-', color=color_lines, alpha=0.1)

    # Plot replicates
    for ax, (var_name, replicates) in zip(axes, dataset.items()):
        props = var_properties.get(var_name, {}) if var_properties else {}
        label = props.get("label", var_name)
        color = props.get("color", None)
        ylabel = props.get("ylabel", label)
        xlabel = props.get("xlabel", label)

        for i, rep in enumerate(replicates):
            t = rep["time"]
            y = rep["values"]
            rep_label = f"{label} (rep {i+1})" if len(replicates) > 1 else label
            ax.plot(t, y, label=rep_label, color=color, lw=line_width, alpha=alpha,
                    marker='o', markersize=4, linestyle='None')

        ax.set_title(label, fontsize=fontsize, fontname=fontname)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        if len(replicates) > 1:
            ax.legend(fontsize=fontsize - 2)

    if suptitle:
        fig.suptitle(suptitle, fontsize=fontsize+2)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()

    return fig, axes