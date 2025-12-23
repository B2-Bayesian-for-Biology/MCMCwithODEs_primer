import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Load posterior
# ------------------------------------------------------------------
file_path = '../res/ehux_monod_everything_mh.nc'
idata = az.from_netcdf(file_path)

posterior = idata.posterior

# Extract samples (flatten chains & draws)
N0 = posterior["N0"].values.flatten()
Qn = posterior["Qn"].values.flatten()

# Remove any NaNs just in case
mask = np.isfinite(N0) & np.isfinite(Qn)
N0 = N0[mask]
Qn = Qn[mask]

print(f"Number of posterior samples used: {len(N0)}")

# ------------------------------------------------------------------
# Fit linear relation: Qn = a * N0 + b
# ------------------------------------------------------------------
X = N0.reshape(-1, 1)
y = Qn

reg = LinearRegression()
reg.fit(X, y)

a = reg.coef_[0]
b = reg.intercept_
r2 = reg.score(X, y)

print("\n=== Linear reparameterization from posterior ===")
print(f"Qn ≈ a * N0 + b")
print(f"a (slope)     = {a:.6e}")
print(f"b (intercept) = {b:.6e}")
print(f"R^2           = {r2:.4f}")

# ------------------------------------------------------------------
# Plot posterior samples + fitted line
# ------------------------------------------------------------------
N0_line = np.linspace(N0.min(), N0.max(), 200)
Qn_line = a * N0_line + b

plt.figure(figsize=(5, 4))
plt.scatter(N0, Qn, s=5, alpha=0.2, label="Posterior samples")
plt.plot(N0_line, Qn_line, color="black", lw=2, label="Best-fit line")

plt.xlabel(r'Init. nutrient concentration, $N_0$ (mmol$/m^3$)')
plt.ylabel( r'Nutrient Quota $Q_N$ (mmol N/cell)')
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Calculate and print the standard deviation of the residuals
# ------------------------------------------------------------------

std_residuals = np.std(Qn - (a * N0 + b))
print(f"Standard deviation of residuals = {std_residuals:.6e}")
