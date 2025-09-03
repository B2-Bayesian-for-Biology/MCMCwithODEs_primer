from legacy import malthusian_normal_plotting as gp
from legacy import death_normal_plotting as dp
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns

# Set up figure
f, ax = plt.subplots(2, 2, figsize=[10, 10])
ax = ax.flatten()

# Axis labels
ax[0].set_xlabel('Time (days)')
ax[0].set_ylabel('Cells (ml$^{-1}$)')

ax[1].set_xlabel('Time (days)')
ax[1].set_ylabel('Cells (ml$^{-1}$)')

ax[2].set_xlabel('Growth rate (day$^{-1}$)')
ax[2].set_ylabel('Probability density')

ax[3].set_xlabel('Growth rate (day$^{-1}$)')
ax[3].set_ylabel('Death rate (day$^{-1}$)')  # assuming delta is death rate

# Posterior predictive plots
gp.plot_posterior_predictive(ax[0], gp.growth_trace, gp.time, gp.obs)
dp.plot_posterior_predictive(ax[1], dp.death_trace, gp.time, gp.obs)

# Extract samples
gdf = az.extract(gp.growth_trace).to_dataframe()
ddf = az.extract(dp.death_trace).to_dataframe()

# KDE plots
az.plot_kde(gdf['mum'].values, ax=ax[2],
            plot_kwargs={'label': 'Growth rate model', 'color': 'g'})
az.plot_kde(ddf['mum'].values, ax=ax[2],
            plot_kwargs={'label': '(Growth-death) rate model', 'color': 'r'})

# Set font size for the labels
ax[2].set_xlabel('Growth rate (day$^{-1}$)', fontsize=10)
ax[2].set_ylabel('Probability density', fontsize=10)
ax[2].tick_params(axis='both', labelsize=10)  # Change tick size for ax[2]


# Scatter plot of growth vs death rates
ax[3].scatter(ddf.mum, ddf.delta, s=10, alpha=0.3, color='red')
# Pair plot of covariance of ddf.mum and ddf.delta
#sns.pairplot(ddf[['mum', 'delta']], diag_kind='kde')

ax[3].set_title('Joint posterior: growth vs death')

# Subplot labels
f.subplots_adjust(wspace=0.3, hspace=0.3)
for (a, l) in zip(ax, 'abcd'):
    a.text(0.05, 0.9, f'({l})', transform=a.transAxes, fontsize=12)



# Set font size for the labels in all subplots
for i in range(len(ax)):
    ax[i].set_xlabel(ax[i].get_xlabel(), fontsize=12)
    ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=12)
    ax[i].tick_params(axis='both', labelsize=12)  # Change tick size for all axes


# Save figure
f.savefig('../figures/main_text_fig2_v2.png', bbox_inches='tight', dpi=600)
plt.close(f)

