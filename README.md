# Real-Time Predictive Coding Compression Evaluation

**Authors:** Athulya Weerakoon, Arosh Upathilak ([GitHub](https://github.com/Arosh-Upathilak))  
**Affiliation:** University of Jaffna, Sri Lanka  
**License:** MIT License (see `LICENSE` file)

---

## Overview

This repository provides **tools and Jupyter notebooks** for evaluating various **real-time predictive coding compression algorithms**, including **regressive, probabilistic, and hybrid approaches**.

The research focuses on:

- Comparing **different predictive coding algorithms** for lossy sensor data compression.  
- Evaluating **compression performance**, reconstruction quality, and computational efficiency.  
- Providing flexible workflows for testing and analyzing algorithms in real-time scenarios.

---

## Repository Contents

- `utils/` – Python utility methods for running experiments and data handling.  
- `notebooks/` – Jupyter notebooks demonstrating example compression workflows and experimental setups.  
- `LICENSE` – MIT License.  
- `pyproject.toml` – Poetry configuration for dependencies.

**Note:** Datasets are **not included** due to size/privacy considerations. Users must provide their own sensor datasets to run the notebooks.

---

## Getting Started

### Requirements

- Python 3.12+  
- [Poetry](https://python-poetry.org/) for dependency management  
- Jupyter Notebook or Jupyter Lab  

### Setup Instructions

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. Install dependencies and build the project with Poetry:

    ```bash
    # Install dependencies
    poetry install

    # Build the package
    poetry build

    # Activate the environment
    poetry shell
    ```

3. Launch a notebook to start experimenting:

    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```

---

## Usage

- Open a notebook from the `notebooks/` folder.  
- Provide your own sensor dataset.  
- Run cells to evaluate different predictive coding compression algorithms.  
- Modify parameters to test regressive, probabilistic, or hybrid approaches.

---

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.