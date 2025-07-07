import arviz as az
import matplotlib.pyplot as plt

idata = az.from_netcdf('../res/vardi_general_chain.nc')

print("Available variables:", list(idata.posterior.data_vars))

Qn = idata.posterior["Qn"].values.flatten()
N0 = idata.posterior["N0"].values.flatten()

plt.figure(figsize=(10, 6))
plt.plot(N0, Qn, marker='.', markersize=5, linestyle=' ', color='k')
plt.tick_params(axis='both', labelsize=14)
plt.title('Qn vs N0', fontsize=16)
plt.xlabel('N0', fontsize=14)
plt.ylabel('Qn', fontsize=14)
plt.title('Qn vs N0')
plt.xlabel('N0')
plt.ylabel('Qn')
plt.grid()
plt.show()