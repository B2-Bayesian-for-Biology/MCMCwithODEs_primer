---
title: 🧬 Baysian Inference for Biological ODEs
---

# 🧬 Baysian Inference for Biological ODEs
### **Bayesian Learning of Microbial Traits from Population Time Series Data: A Primer**

Mathematical models are increasingly used to infer traits, interactions, and functional dynamics of microbial systems. The inference process typically begins with a rate-based **Ordinary Differential Equation (ODE)** model.

However, fitting such models to experimental data requires a principled statistical framework that can:
> Incorporate prior knowledge,
> Account for measurement noise, and
> Quantify uncertainty in parameter estimates.

!!! quote "Our aim"
    We strive to make the **implicit, explicit** — introducing **Bayesian inference of ecological ODE models** for microbial time series, with a unified workflow in **Python** (PyMC) and **Julia** (Turing.jl).

---

## ⬇️ Download the full repository

Get all notebooks, code, and examples in one click:

<p align="center">
  <a href= "https://github.com/B2-Bayesian-for-Biology/MCMCwithODEs_primer/archive/refs/heads/main.zip">
    <strong>📦 Download ZIP</strong>
  </a>
</p>

---

## 📘 Overview

This repository accompanies our upcoming paper:

> **“Bayesian Learning of Microbial Traits from Population Time Series Data: A Primer”**  
> *Authors:* Raunak Dey, Robert Beach, Kennedi M. Hambrick, Ioannis Sgouralis, Paul Fremont, David Demory, Eric Carr, Stephen J. Beckett, Joshua S. Weitz, David Talmy
> *(Link will be posted here when the paper is online.)*

---

## 🚀 Case studies

### Python (PyMC) 

You can view the solved examples (case studies) or try running it yourself via Google Colab.

- **Case Study 1 — Exponential Growth and Death**  
  📄 [View notebook on GitHub](education/python/case_study_1.ipynb)  
  🧑‍💻 Try it yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B2-Bayesian-for-Biology/MCMCwithODEs_primer/blob/main/Colab/python/case_study_1.ipynb)

- **Case Study 2 — Logistic Growth and Death**  
  📄 [View notebook on GitHub](education/python/case_study_2.ipynb)  
  🧑‍💻 Try it yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B2-Bayesian-for-Biology/MCMCwithODEs_primer/blob/main/Colab/python/case_study_2.ipynb)

- **Case Study 3 — Monod Growth and Death**  
  📄 [View notebook on GitHub](education/python/case_study_3.ipynb)  
  🧑‍💻 Try it yourself: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/B2-Bayesian-for-Biology/MCMCwithODEs_primer/blob/main/Colab/python/case_study_3.ipynb)

### Julia (Turing.jl)

- [Case Study 1 — Exponential Growth and Death](education/julia/case_study_1.ipynb)
- [Case Study 2 — Logistic Growth and Death](education/julia/case_study_2.ipynb)
- [Case Study 3 — Monod Growth and Death](education/julia/case_study_3.ipynb)
---

## 🧰 Stack

- **Python · [PyMC](https://www.pymc.io/)** — Probabilistic programming in Python for Bayesian modeling and inference  
- **Julia · [Turing.jl](https://turinglang.org/)** — A flexible probabilistic programming language in Julia  

All analyses were performed using:

**Python**
- Python 3.12  
- PyMC 5.25.1  
- PyTensor 2.31.7  
- ArviZ 0.22.0  
- NumPy 2.2.5  
- SciPy 1.16.2  

**Julia**
- Julia 1.11.0  
- Turing.jl 0.40.2  
- DifferentialEquations.jl 7.16.1  
- SciMLSensitivity.jl 7.90.0  
- Distributions.jl 0.25.120  
- MCMCChains.jl 7.2.0  

---

## 👩‍🔬 Contributors

This primer was developed through a collaborative effort across multiple research groups, combining expertise in **microbial ecology**, **statistical physics**, and **Bayesian inference**.

**Raunak Dey** — University of Maryland  
🔗 [Website](https://raunakdey.github.io) · [GitHub](https://github.com/RaunakDey)

**David Talmy** — University of Tennessee, Knoxville  
🔗 [Website](https://eeb.utk.edu/people/david-talmy/) 

**Robert Beach** — University of Tennessee, Knoxville  

**Kennedi Hambrick** — University of Tennessee, Knoxville  

**Ioannis Sgouralis** — University of Tennessee, Knoxville  
🔗 [Website](https://math.utk.edu/labs/sgouralis/)

**Stephen J. Beckett** —  University of Maryland  🔗 [Website](https://sjbeckett.github.io)


**Paul Frémont** —  University of Maryland 
🔗 [Website](https://www.paulfremont.com)

**David Demory** — Sorbonne Université 
🔗 [Website](https://usr3579.obs-banyuls.fr/fr/axe-genophy/membres-permanents/daviddemory.html)


**Eric Carr** — University of Tennessee, Knoxville  

**Joshua S. Weitz** — University of Maryland 
🔗 [Website](https://weitzgroup.umd.edu/)  

---

> This project connects **theory, data, and computation** to advance reproducible Bayesian inference for ecological population models.