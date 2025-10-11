# 🧬 MCMCwithODEs_primer  
### **Bayesian Learning of Microbial Traits from Population Time Series Data: A Primer**

Mathematical models are increasingly used to infer traits, interactions, and functional dynamics of microbial systems. The inference process typically begins with the development of a rate-based **Ordinary Differential Equation (ODE)** model.  

However, fitting such models to experimental data requires a principled statistical framework that can:  
- Incorporate prior knowledge,  
- Account for measurement noise, and  
- Quantify uncertainty in parameter estimates.  

Such principles are often *assumed to be understood implicitly*. Here, we strive to make the **implicit, explicit**.  

This **primer** introduces **Bayesian inference of ecological ODE models** for microbial time series, with three detailed case studies of algal population dynamics governed by a **birth–death process**.  

Through this project, we connect **theory, code, and data** using a unified Bayesian framework implemented in both **Python** (via [PyMC](https://www.pymc.io)) and **Julia** (via [Turing.jl](https://turinglang.org)).  
We hope this resource helps bring the utility of **Bayesian learning** to the broader **microbial ecology** and **quantitative biology** communities.

---

## 📘 Overview

This repository accompanies our upcoming paper:

> **“Bayesian Learning of Microbial Traits from Population Time Series Data: A Primer”**  
> *Authors:* TBD
> *(Link will be posted here when the paper is online.)*

The repository contains:
- Example ODE-based ecological models  
- Step-by-step **MCMC tutorials** in both PyMC (Python) and Turing.jl (Julia)  
- Comparison of deterministic vs Bayesian approaches  
- Jupyter notebooks and Julia scripts demonstrating model calibration, posterior estimation, and uncertainty quantification  

---

## 🧩 Example: Bayesian Inference Workflow

Below is an example visualization from the primer, illustrating Bayesian inference applied to microbial population dynamics:

<img width="800" alt="Bayesian Inference Example" src="https://github.com/user-attachments/assets/19d341b8-0596-428b-be9b-648b5f75f9ce" />

---

## 🧠 Key Concepts

- **Model definition:** Rate-based ODEs representing microbial birth–death or interaction dynamics  
- **Likelihood formulation:** Normal or Log-Nornal models for time series data  
- **Prior specification:** Informative or weakly-informative priors on biological parameters  
- **Posterior inference:** Using MCMC/HMC sampling via PyMC or Turing  
- **Cross-platform reproducibility:** Equivalent inference workflows in both Python and Julia  

---


### Examples
```bash
pip install pymc arviz numpy scipy matplotlib
jupyter notebook education/python/case_study_1.ipynb
```

### Citation

To be added.




