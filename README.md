[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17792668.svg)](https://doi.org/10.5281/zenodo.17792668)

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


## ▶️ Try it yourself (Google Colab)

You can view the solved examples (case studies) or try running it yourself via Google Colab directly in your browser — no local setup required.

- **Case Study 1 — Exponential Growth and Death**  
  📄 [View notebook on GitHub](education/python/case_study_1.ipynb)  
  🧑‍💻 Try it yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B2-Bayesian-for-Biology/MCMCwithODEs_primer/blob/main/Colab/python/case_study_1.ipynb)

- **Case Study 2 — Logistic Growth and Death**  
  📄 [View notebook on GitHub](education/python/case_study_2.ipynb)  
  🧑‍💻 Try it yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B2-Bayesian-for-Biology/MCMCwithODEs_primer/blob/main/Colab/python/case_study_2.ipynb)

- **Case Study 3 — Monod Growth and Death**  
  📄 [View notebook on GitHub](education/python/case_study_3.ipynb)  
  🧑‍💻 Try it yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B2-Bayesian-for-Biology/MCMCwithODEs_primer/blob/main/Colab/python/case_study_3.ipynb)

The Colab version automatically clones the repository and adds the project root
to the Python path so that shared utilities in `utils/` are available.

--- 

## 🧪 Run the notebooks locally

This repository is a research and educational codebase (not a packaged Python library).  
To run the example notebooks, please follow the steps below.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/B2-Bayesian-for-Biology/MCMCwithODEs_primer.git
cd MCMCwithODEs_primer
```
### 2️⃣ Create and activate a Python environment
```python
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```
### 3️⃣ Install dependencies
```bash
pip install --upgrade pip
pip install -r python_requirements.txt
```
Note: The notebooks were tested with the package versions listed in python_requirements.txt. Other versions may work but are not guaranteed.

### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```
### 5️⃣ Open and run a case study
Navigate to:
```text
docs/education/python/case_study_2.ipynb
```
and run the notebook top to bottom. This notebook demonstrates Bayesian inference of microbial growth and death
rates from population time-series data using PyMC and a Logistic growth model.

## 🧩 Example: Bayesian Inference Workflow

Below is an example visualization from the primer, illustrating Bayesian inference applied to microbial population dynamics:

<img width="800" alt="Bayesian Inference Example" src="https://github.com/user-attachments/assets/19d341b8-0596-428b-be9b-648b5f75f9ce" />

---


## 📈 Logistic Growth Example

A simple example used in the primer is the **logistic growth model**, a foundational ecological model describing population growth with a carrying capacity.

### Mathematical Model

The logistic growth equation is given by:

$$ \frac{dP}{dt} = rP \left(1 - \frac{P}{K}\right) $$

where:  
- \( P(t) \) = population size at time \( t \)  
- \( r \) = intrinsic growth rate  
- \( K \) = carrying capacity  

---

### Python Implementation

```python
def logistic_growth(y, t, params):

    P = y[0]
    r = params[0]
    K = params[1]
    dydt[0] = r * (1 - P / K) * P 
    
    return dydt
```

## Example Outputs

### Simulated Dynamics

Population growth over time under logistic dynamics:
<img width="800" alt="Logistic Growth Dynamics" src="https://github.com/dtalmy/MCMCwithODEs_primer/blob/main/case_study_2/python/figures/vardi_logistic_growth_dynamics_corrected.png" />


## MCMC Posterior Chains

### MCMC chains showing convergence for inferred parameters \( r \)  and \( K \) :

<img width="800" alt="Logistic Growth Chains" src="https://github.com/dtalmy/MCMCwithODEs_primer/blob/main/case_study_2/python/figures/vardi_logistic_growth_chains_corrected.png" />



---
## 🧠 Key Concepts

- **Model definition:** Rate-based ODEs representing microbial birth–death or interaction dynamics  
- **Likelihood formulation:** Normal or Log-Nornal models for time series data  
- **Prior specification:** Informative or weakly-informative priors on biological parameters  
- **Posterior inference:** Using MCMC/HMC sampling via PyMC or Turing  
- **Cross-platform reproducibility:** Equivalent inference workflows in both Python and Julia  

---

### Citation

To be added.




