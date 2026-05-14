"""
identifiability_prior_mean_sweep.py

Companion to identifiability_track.py.

What this does
--------------
Fixes the prior *width* on δ (sigma_delta = 0.1, i.e. moderately informative)
and sweeps the prior *mean* of δ across a range.  Because dP/dt = (μ-δ)P,
the data only constrain (μ-δ); a wrong prior mean on δ will bias the posterior
mean of μ by a near-equal amount.  This directly answers the reviewer question:
"how wrong does a prior need to be to bias the output?"

Outputs
-------
- One .nc trace file per prior mean value  (trace_mean_delta_<val>.nc)
- identifiability_prior_mean_sweep.svg     (the new supplementary figure)
- Console table: prior mean δ → posterior mean μ, posterior std μ
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import arviz as az
import pymc as pm

# ---------------------------------------------------------------------------
# ODE + model builder  (same structure as identifiability_track.py)
# ---------------------------------------------------------------------------

def cells_ode(y, t, params):
    mum, delta = params[0], params[1]
    return [(mum - delta) * y[0]]


def build_model(data, obs, prior_mean_delta, sigma_delta=0.1):
    """Build PyMC model with a given prior mean on δ."""
    cell_model = pm.ode.DifferentialEquation(
        func=cells_ode,
        times=data['times'].values,
        n_states=1,
        n_theta=2,
        t0=0,
    )
    with pm.Model() as model:
        mum   = pm.TruncatedNormal('mum',   mu=0.5,              sigma=0.3,
                                   lower=0.01, upper=1.5)
        delta = pm.TruncatedNormal('delta', mu=prior_mean_delta,  sigma=sigma_delta,
                                   lower=0.01, upper=1.5)
        N0    = pm.Lognormal('N0', mu=np.log(obs[0]), sigma=0.1)
        sigma = pm.HalfNormal('sigma', 1)

        y_hat = cell_model(y0=[N0], theta=[mum, delta])
        pm.Normal('Y_obs',
                  mu=pm.math.log(y_hat[:, 0]),
                  sigma=sigma,
                  observed=np.log(obs))
    return model


def run_inference(model, draws=2000, tune=2000, chains=2, target_accept=0.95):
    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          return_inferencedata=True,
                          target_accept=target_accept)
    return trace


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- load data --------------------------------------------------------
    data = pd.read_csv("./../data/phaeocystis_control.csv")
    time = data['times'].values
    obs  = data['cells'].values

    # ---- sweep parameters -------------------------------------------------
    # Fixed: moderately informative prior width (same as panel c in S3)
    SIGMA_DELTA = 0.1

    # Sweep prior mean of δ from well below to well above the effective growth rate.
    # Adjust this range to bracket whatever the data-implied net rate is.
    prior_mean_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    posterior_means = []
    posterior_stds  = []
    traces          = {}

    for pm_delta in prior_mean_values:
        print(f"\n=== Prior mean of δ = {pm_delta} ===")
        model = build_model(data, obs,
                            prior_mean_delta=pm_delta,
                            sigma_delta=SIGMA_DELTA)
        trace = run_inference(model, draws=2000, tune=2000, chains=4)

        # save
        fname = f"./../data/trace_mean_delta_{pm_delta}.nc"
        az.to_netcdf(trace, fname)
        traces[pm_delta] = trace

        # summary stats
        mum_post = trace.posterior['mum'].values.flatten()
        mu_mean  = float(np.mean(mum_post))
        mu_std   = float(np.std(mum_post))
        posterior_means.append(mu_mean)
        posterior_stds.append(mu_std)
        print(f"  Posterior μ:  mean = {mu_mean:.4f},  std = {mu_std:.4f}")

    # ---- print table ------------------------------------------------------
    print("\n" + "="*55)
    print(f"{'Prior mean δ':>14} | {'Post mean μ':>12} | {'Post std μ':>11}")
    print("-"*55)
    for pmd, pm_, ps in zip(prior_mean_values, posterior_means, posterior_stds):
        print(f"{pmd:>14.2f} | {pm_:>12.4f} | {ps:>11.4f}")

    # ---- figure -----------------------------------------------------------
    #
    # Layout: two rows
    #   Row 1 (top): posterior histograms of μ, one panel per prior mean of δ
    #   Row 2 (bottom-left): posterior mean of μ vs prior mean of δ  (scatter)
    #   Row 2 (bottom-right): posterior std  of μ vs prior mean of δ (scatter)
    #
    n_panels = len(prior_mean_values)
    ncols    = n_panels          # one column per prior-mean value
    fig      = plt.figure(figsize=(3.5 * ncols, 9))
    gs       = gridspec.GridSpec(2, ncols, figure=fig,
                                 hspace=0.45, wspace=0.35)

    # colour map: cold→warm as prior mean of δ increases (misleading → well-specified)
    colours = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_panels))

    # -- top row: histograms ------------------------------------------------
    for col_idx, (pmd, colour) in enumerate(zip(prior_mean_values, colours)):
        ax = fig.add_subplot(gs[0, col_idx])
        trace = traces[pmd]
        mum_post = trace.posterior['mum'].values.flatten()

        ax.hist(mum_post, bins=40, range=(0, 1.5),
                density=True, color=colour, alpha=0.75,
                edgecolor='none')

        # prior curve for δ  (for reference)
        x = np.linspace(0.01, 1.5, 400)
        delta_dist = pm.TruncatedNormal.dist(
            mu=pmd, sigma=SIGMA_DELTA, lower=0.01, upper=1.5)
        prior_pdf = np.exp(pm.logp(delta_dist, x).eval())
        ax.plot(x, prior_pdf, color='black', linewidth=2, label='Prior of δ')

        # vertical line at posterior mean of μ
        ax.axvline(posterior_means[col_idx], color=colour,
                   linewidth=2, linestyle='--', alpha=0.9)

        label = f"μ_δ = {pmd}"
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("μ and δ (/day)", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("PDF", fontsize=11)
        ax.set_xlim(0, 1.5)
        ax.tick_params(labelsize=10)

    # add a shared legend to the last top panel
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor='grey', alpha=0.6, label='Posterior of μ'),
        Line2D([0], [0], color='black', linewidth=2, label='Prior of δ'),
        Line2D([0], [0], color='grey', linewidth=2,
               linestyle='--', label='Post. mean of μ'),
    ]
    fig.add_subplot(gs[0, -1]).set_visible(False)   # hide; legend placed manually
    fig.legend(handles=legend_handles, loc='upper right',
               bbox_to_anchor=(0.99, 0.97), fontsize=11, frameon=True)

    # -- bottom-left: posterior mean of μ vs prior mean of δ ----------------
    ax_mean = fig.add_subplot(gs[1, :ncols//2])
    ax_mean.scatter(prior_mean_values, posterior_means,
                    color=[c for c in colours], s=80, zorder=3, edgecolor='k')

    # 1:1 reference line  (if prior were unconstrained, post mean ≈ net rate + prior mean δ)
    xline = np.array([min(prior_mean_values), max(prior_mean_values)])
    # offset the 1:1 line by the apparent net growth rate
    # (posterior mean when prior mean δ is smallest ≈ μ_net + δ_prior_min)
    net_rate_approx = posterior_means[0] - prior_mean_values[0]
    ax_mean.plot(xline, xline + net_rate_approx,
                 'k--', linewidth=1.5, label=f'Slope-1 ref\n(net rate ≈ {net_rate_approx:.2f})')

    ax_mean.set_xlabel("Prior mean of δ (/day)", fontsize=13)
    ax_mean.set_ylabel("Posterior mean of μ (/day)", fontsize=13)
    ax_mean.set_title("Bias in μ from misspecified δ prior", fontsize=13)
    ax_mean.legend(fontsize=11)
    ax_mean.tick_params(labelsize=11)

    # -- bottom-right: posterior std of μ vs prior mean of δ ----------------
    ax_std = fig.add_subplot(gs[1, ncols//2:])
    ax_std.scatter(prior_mean_values, posterior_stds,
                   color=[c for c in colours], s=80, zorder=3, edgecolor='k')
    ax_std.set_xlabel("Prior mean of δ (/day)", fontsize=13)
    ax_std.set_ylabel("Posterior std of μ (/day)", fontsize=13)
    ax_std.set_title("Posterior uncertainty in μ\nvs prior mean of δ", fontsize=13)
    ax_std.tick_params(labelsize=11)

    fig.suptitle(
        "Effect of prior mean of δ on inferred μ  (σ_δ fixed = 0.1)",
        fontsize=14, y=1.01
    )

    plt.savefig("./../data/identifiability_prior_mean_sweep.svg",
                bbox_inches='tight')
    plt.savefig("./../data/identifiability_prior_mean_sweep.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nFigure saved.")