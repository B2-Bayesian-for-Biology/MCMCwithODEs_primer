import malthusian_plotting as gp
import malthusian_death_plotting as dp
import matplotlib.pyplot as plt
import arviz as az

################################
# load data and setup figs
################################

# setup figure
f,ax = plt.subplots(2,2,figsize=[10,10])

ax = ax.flatten()
ax[0].set_xlabel('Time (days)')
ax[1].set_xlabel('Time (days)')
ax[2].set_xlabel('Growth rate (day$^{-1}$)')
ax[3].set_xlabel('Growth rate (day$^{-1}$)')
ax[0].set_ylabel('Cells (ml$^{-1}$')
ax[1].set_ylabel('Cells (ml$^{-1}$')
ax[2].set_ylabel('Probability density (day$^{-1}$)')
ax[3].set_ylabel('Growth rate (day$^{-1}$)')

################################
# dynamics
################################

gp.plot_posterior_predictive(ax[0], gp.growth_trace, gp.time, gp.obs)
dp.plot_posterior_predictive(ax[1], dp.death_trace, gp.time, gp.obs)

################################
# kernel plots
################################

gdf = az.extract(gp.growth_trace).to_dataframe()
ddf = az.extract(dp.death_trace).to_dataframe()

ax[2].hist(gdf.mum,alpha=0.5,density=True,color='r',label='Growth rate model')
ax[2].hist(ddf.mum,alpha=0.5,density=True,color='g',label='(Growth-death) rate model')

l2 = ax[2].legend()
l2.draw_frame(False)

################################
# scatter plot
################################

ax[3].scatter(ddf.mum,ddf.delta)

f.subplots_adjust(wspace=0.3,hspace=0.3)
for (a,l) in zip(ax,'abcd'):
    a.text(0.1,0.9,l,transform=a.transAxes)
f.savefig('../figures/main_text_fig2',bbox_inches='tight',dpi=300)

plt.close(f)


