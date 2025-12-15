import arviz as az
import numpy as np

def summarize_trace_table(nc_path):
    idata = az.from_netcdf(nc_path)
    return az.summary(
        idata,
        var_names=None,
        stat_funcs={"mean": np.mean, "std": np.std},
        extend=True
    )[["mean", "std"]]

def summarize_trace_nc(nc_path):
    """
    Load a PyMC/ArviZ .nc trace and print mean and variance
    for each posterior parameter (across chains & draws).
    """
    idata = az.from_netcdf(nc_path)

    posterior = idata.posterior

    print(f"\nLoaded trace from: {nc_path}")
    print(f"Chains: {posterior.sizes.get('chain', 1)}")
    print(f"Draws per chain: {posterior.sizes.get('draw', 'unknown')}\n")

    for var in posterior.data_vars:
        values = posterior[var].values  # shape: (chain, draw, ...)

        flat = values.reshape(-1, *values.shape[2:])  # combine chain+draw
        mean = np.mean(flat, axis=0)
        std_ = np.std(flat, axis=0)

        print(f"Parameter: {var}")
        print(f"  Mean: {mean}")
        print(f"  Std: {std_}\n")

