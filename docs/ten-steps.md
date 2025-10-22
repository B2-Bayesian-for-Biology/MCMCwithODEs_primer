---
title: Mini Tutorial — 10 Steps of Bayesian Inference
---

# 🧭 Mini Tutorial: 10 Steps of Bayesian Inference

Individual Bayesian inverse modeling is case-specific, but this page summarizes the **common workflow** we follow in the primer. We illustrate with simple population models (exponential, logistic, and a Monod resource model).

> **Notation.** Data $\mathcal{D}$, parameters $\theta$, likelihood $\mathcal{L}(\mathcal{D}\mid\theta)$, prior $P(\theta)$, posterior $P(\theta\mid\mathcal{D})$.

---

???+ summary "1) Visualize the data"
    Begin by plotting time series for each species/variable.  
    - Use **log-scaled y-axes** when densities span orders of magnitude.  
    - Early visualization guides **model choice** and highlights measurement scales and noise structure.

???+ summary "2) Choose a dynamical model"
    Qualitative trends inform the structure of your ODEs.

    **Exponential growth & death (Malthusian):**
    $$
    \frac{dP}{dt}=\mu P, \qquad
    \frac{dP}{dt}=(\mu-\delta)P
    $$
    where $P$ is cell density, $\mu$ growth rate, $\delta$ death rate.

    **Logistic growth (with death):**
    $$
    \frac{dP}{dt}=rP\!\left(1-\frac{P}{K}\right)-\delta P, \qquad
    \frac{dD}{dt}=\delta P
    $$
    with intrinsic rate $r$, carrying capacity $K$, live cells $P$, dead cells $D$.

    **Monod (resource-explicit) growth with death:**
    $$
    \begin{aligned}
    \frac{dN}{dt}&=-\frac{Q_N\,\mu_{\max}N}{N+K_S}\,\big(P\times10^6\big) \\\\
    \frac{dP}{dt}&=\frac{\mu_{\max}N}{N+K_S}\,P-\delta P \\\\
    \frac{dD}{dt}&=\delta P
    \end{aligned}
    $$
    where $N$ is nutrient, $Q_N$ cellular quota, $\mu_{\max}$ max growth rate, and $K_S$ half-saturation.

???+ summary "3) Define the likelihood"
    The likelihood $\mathcal{L}(\mathcal{D}\mid\theta)$ measures compatibility of model and data.  
    - Common choice: **Gaussian errors** (possibly on a log scale) with equal weights across observations.  
    - When variables are **latent** (unobserved), build the likelihood only over observed variables.  
    - With equal weights, $\log\mathcal{L}$ reduces to **sum of squared errors** plus noise terms.

???+ summary "4) Select parameters to fit"
    Include dynamic parameters, initial conditions, and observation noise.  
    - For strong **posterior covariance**, consider **reparameterization** and fit independent parameters only.  
    - Keep units and identifiability in mind.

???+ summary "5) Choose priors"
    Use literature, physics, and bounds to set priors.  
    - **Normal / Log-Normal** for parameters spanning orders of magnitude.  
    - **Uniform** for weakly informative ranges.  
    - **Half-Normal** (or Half-Cauchy) for **positive** noise scales.  
    - Apply **truncation** to enforce positivity (e.g., growth/death rates).

???+ summary "6) Pick sampler and ODE solver"
    - For smooth models with gradients, **HMC/NUTS** is efficient; needs Jacobians/AD support.  
    - For **stiff ODEs** or tricky gradients, **Metropolis** (or other gradient-free samplers) with robust solvers (e.g., `solve_ivp`) can be preferable.  
    - In PyMC, you can use `DifferentialEquation` for coupled ODE integration within sampling.

???+ summary "7) Sample the posteriors"
    Bayes’ rule:
    $$
    P(\theta\mid\mathcal{D}) \propto \mathcal{L}(\mathcal{D}\mid\theta)\,P(\theta)
    $$
    Practical tips:
    - **Prior predictive checks** to validate priors before seeing data.  
    - **Warm-up/burn-in** long enough for adaptation (more for high-dim models).  
    - **Draws & chains**: target $\ge$ 1000 effective samples per parameter.  
    - **Acceptance rate**: tune as needed (e.g., PyMC default ~0.8; can increase toward 0.95 if convergence requires it).

???+ summary "8) Inspect posterior distributions"
    - Use **marginals** and **corner plots** to assess shape, multimodality, and correlations.  
    - Strong correlations → consider **reparameterization** or model refinement.  
    - Report credible intervals and posterior summaries with context.

???+ summary "9) Convergence diagnostics"
    Ensure reliable exploration before interpreting results:  
    - **$\hat R$ (Gelman–Rubin)** close to 1 (e.g., < 1.01).  
    - **Autocorrelation** decays quickly (e.g., < 0.1 by ~50 lags).  
    - **ESS (bulk & tail)** sufficiently large (e.g., > 300).

???+ summary "10) Posterior predictive simulation"
    Propagate posterior draws through the ODE to generate **predictive trajectories**.  
    - Compare to data for **model fit**.  
    - Use spread to **quantify predictive uncertainty** in dynamics.