import arviz as az
import matplotlib.pyplot as plt

idata = az.from_netcdf('../res/vardi_general_chain.nc')

print("Available variables:", list(idata.posterior.data_vars))

Qn = idata.posterior["Qn"].values.flatten()
N0 = idata.posterior["N0"].values.flatten()

plt.figure(figsize=(8,8))
plt.plot(N0, Qn, marker='.', markersize=5, linestyle=' ', color='k')
plt.tick_params(axis='both', labelsize=17)
#plt.title('Qn vs N0', fontsize=16)
plt.xlabel('N0', fontsize=18)
plt.ylabel('Qn', fontsize=18)
#plt.title('Qn vs N0')


plt.xlabel(r'$N_0$ (mmol N/cell)')
plt.ylabel(r'$Q_N$ (mmol N/cell)')

plt.grid()
plt.show()
plt.savefig('Qn_vs_N0.svg')