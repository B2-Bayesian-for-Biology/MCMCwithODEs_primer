import matplotlib.pyplot as plt
import numpy as np

def simple_plot_trace_with_priors(idata, var_names, priors, bins=60, force_sci=False):
    try:
        from scipy.stats import gaussian_kde
        use_kde = True
    except Exception:
        use_kde = False

    fig, axes = plt.subplots(len(var_names), 2, figsize=(10, 2*len(var_names)), squeeze=False)

    for i, var in enumerate(var_names):
        arr = idata.posterior[var].values  # (chains, draws, ...)
        chains, draws = arr.shape[:2]
        vals = arr.reshape(chains*draws, -1)[:, 0]
        vals = vals[np.isfinite(vals)]
        axL, axR = axes[i]

        # ---- Left: posterior density + SciPy prior
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        lo, hi = vmin, vmax

        for spec in priors.get(var, []):
            dist = spec["dist"]
            # widen range using prior quantiles (safe if huge/small)
            try:
                qlo, qhi = dist.ppf(0.001), dist.ppf(0.999)
                if np.isfinite(qlo): lo = min(lo, float(qlo))
                if np.isfinite(qhi): hi = max(hi, float(qhi))
            except Exception:
                pass

        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = vmin, vmax
        if lo == hi:
            pad = 1.0 if hi == 0 else abs(hi)*0.05
            lo, hi = hi - pad, hi + pad
        grid = np.linspace(lo, hi, 400)
        

        # posterior: one PDF per chain (no averaging)
        y_top = 1e-12
        for c in range(chains):
            chain_vals = arr[c, :, ...].reshape(draws, -1)[:, 0]
            chain_vals = chain_vals[np.isfinite(chain_vals)]
            if use_kde and chain_vals.size > 1:
                kd = gaussian_kde(chain_vals)
                y = kd(grid)
                axL.plot(grid, y, linewidth=1.3, alpha=0.9, label=f"Post {c+1}")
                y_top = max(y_top, float(np.nanmax(y)))
            else:
                n, _, _ = axL.hist(
                    chain_vals, bins=bins, density=True,
                    histtype="step", linewidth=1.2, alpha=0.9, label=f"Post {c+1}"
                )
                nmax = np.nanmax(n)
                if np.isfinite(nmax): y_top = max(y_top, float(nmax))

        axL.set_ylim(0, y_top * 1.10)

        # priors
        for spec in priors.get(var, []):
            dist = spec["dist"]
            label = spec.get("label", "Prior")
            ypri = dist.pdf(grid)
            axL.plot(grid, ypri, "--", linewidth=2, label=label, color='black', zorder=3)
            y_top = max(y_top, float(np.nanmax(ypri)))

        axL.set_ylim(0, y_top*1.10)
        axL.set_xlabel(var)
        axL.set_ylabel("Density")
        axL.set_title(f"{var} — Posterior PDF (+ Prior)")
        if priors.get(var): axL.legend(loc="best", frameon=False)

        # optional: force scientific notation on left axes
        if force_sci:
            from matplotlib.ticker import ScalarFormatter

            # X axis formatter
            fmtx = ScalarFormatter(useMathText=True)
            fmtx.set_scientific(True)
            fmtx.set_powerlimits((-3, 4))
            fmtx.set_useOffset(False)
            axL.xaxis.set_major_formatter(fmtx)
            #axL.get_xaxis().get_offset_text().set_visible(False)

            # Y axis formatter (separate instance!)
            fmty = ScalarFormatter(useMathText=True)
            fmty.set_scientific(True)
            fmty.set_powerlimits((-3, 4))
            fmty.set_useOffset(False)
            axL.yaxis.set_major_formatter(fmty)
            #axL.get_yaxis().get_offset_text().set_visible(False)

            # X axis (right plot)
            fmtxR = ScalarFormatter(useMathText=True)
            fmtxR.set_scientific(True)
            fmtxR.set_powerlimits((-3, 4))   # always use ×10^n
            fmtxR.set_useOffset(False)      # no '1e6' banner
            axR.xaxis.set_major_formatter(fmtxR)
            #axR.get_xaxis().get_offset_text().set_visible(False)

            # Y axis (right plot)
            fmtyR = ScalarFormatter(useMathText=True)
            fmtyR.set_scientific(True)
            fmtyR.set_powerlimits((-3, 4))
            fmtyR.set_useOffset(False)
            axR.yaxis.set_major_formatter(fmtyR)
            #axR.get_yaxis().get_offset_text().set_visible(False)


        # ---- Right: trace
        x = np.arange(draws)
        series = arr.reshape(chains, draws, -1)[:, :, 0]
        for c in range(chains):
            axR.plot(x, series[c], alpha=0.8)
        axR.set_xlabel("Draw"); axR.set_ylabel(var); axR.set_title(f"{var} — Trace (by chain)")

    fig.tight_layout()
    return fig, axes
