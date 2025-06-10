import matplotlib.pyplot as plt
import arviz as az

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





