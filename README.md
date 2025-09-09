# Spline-Based Bayesian Predictive Coding with ANS for Lossy Sensor Data Compression

**Authors:** Athulya Weerakoon, Arosh Upathilak ([GitHub](https://github.com/Arosh-Upathilak))  
**Affiliation:** University of Jaffna, Sri Lanka  
**License:** MIT License (see `LICENSE` file)

---

## Overview

This repository contains utility methods and Jupyter notebooks used in our undergraduate research project:

**"Spline-Based Bayesian Predictive Coding with Asymmetric Numeral Systems (ANS) for Lossy Sensor Data Compression"**

The research explores:

- Using **B-splines** to model probability distributions for Bayesian Predictive Coding (BPC).  
- Applying **Asymmetric Numeral Systems (ANS)** for efficient entropy encoding.  
- Fitting posterior distributions using **S-ADVI (Stochastic Automatic Differentiation Variational Inference)**.  
- Evaluating compression performance on sensor data against baselines.

---

## Repository Contents

- **`utils/`** – Python utility methods used across test setups.  
- **`notebooks/`** – Jupyter notebooks with test environments, example compression workflows, and experimental setups.  
- **`LICENSE`** – MIT License.  
- **`pyproject.toml`** – Poetry configuration for dependencies.

**Note:** Datasets used in the experiments are **not included** in this repository due to size/privacy considerations. Users must provide their own sensor datasets to run the notebooks.

---

## Getting Started

### Requirements

- Python 3.12+  
- [Poetry](https://python-poetry.org/) (recommended for managing dependencies)  
- Jupyter Notebook or Jupyter Lab  

### Install dependencies

```bash
# Install dependencies via Poetry
poetry install

# Activate the environment
poetry shell