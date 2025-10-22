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

## 📘 Overview

This repository accompanies our upcoming paper:

> **“Bayesian Learning of Microbial Traits from Population Time Series Data: A Primer”**  
> *Authors:* TBD  
> *(Link will be posted here when the paper is online.)*

---

**New to the workflow?** See the [Mini Tutorial — 10 Steps](ten-steps.md) for a concise, expandable walkthrough.

## 🚀 Quick start

**Explore the case studies** from the sidebar:
- **Python (PyMC)** → Case Study 1–3  
- **Julia (Turing.jl)** → Case Study 1–2

Or jump directly:

- [Case Study 1 — Exponential Growth and Death](education/python/case_study_1.ipynb)
- [Case Study 2 — Logistic Growth and Death](education/python/case_study_2.ipynb)
- [Case Study 3 — Monod Growth and Death](education/python/case_study_3.ipynb)

---

## 🧰 Stack

- **Python · [PyMC](https://www.pymc.io/)** — Probabilistic programming in Python for Bayesian modeling and inference  
- **Julia · [Turing.jl](https://turinglang.org/)** — A flexible probabilistic programming language in Julia  


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

**Stephen J. Beckett** —  University of Maryland 

**Paul Frémont** —  University of Maryland 

**David Demory** — CNRS / Université Paris-Saclay  

**Eric Carr** — University of Tennessee, Knoxville  

**Joshua S. Weitz** — University of Maryland 
🔗 [Website](https://weitzgroup.umd.edu/)  

---

> This project connects **theory, data, and computation** to advance reproducible Bayesian inference for ecological population models.