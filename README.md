# Real-Time Predictive Coding Compression Evaluation

**Authors:** Athulya Weerakoon, Arosh Upathilak ([GitHub](https://github.com/Arosh-Upathilak))  
**Affiliation:** University of Jaffna, Sri Lanka  
**License:** MIT License (see `LICENSE` file)

## Overview

This repository contains predictive-coding compression tools and research notebooks. The [`notebooks/kan-poc.ipynb`](notebooks/kan-poc.ipynb), [`notebooks/mlp-poc.ipynb`](notebooks/mlp-poc.ipynb), and [`notebooks/lr-poc.ipynb`](notebooks/lr-poc.ipynb) notebooks compare fixed one-step predictors, signed residual ticks, and ANS coding on the Air Temperature series. All three are proofs of concept, not optimized deployment models. They import reusable study helpers from `compression_evaluation.poc` (and plotting helpers from `compression_evaluation.poc_plots`).

The six earlier root notebooks are preserved byte-for-byte under [`notebooks/old_notebooks/`](notebooks/old_notebooks/): `bpc-with-ans.ipynb`, `lpc-with-ans.ipynb`, `mtkan-with-ans-2-precisions copy.ipynb`, `mtkan-with-ans.ipynb`, `neural-network.ipynb`, and `tans.ipynb`. They are archived, not active entry points. Their original relative dataset/import paths may no longer resolve from the archive directory; adjust paths if running them. The two previously archived notebooks remain there unchanged. Checkpoints and image files are not part of this notebook reorganization.

## Dataset and environment

Datasets are not included. All three notebooks expect `datasets/AriviyalN_Data_Air_Temperature.csv` at the repository root. They require Python with NumPy, pandas, matplotlib, and this local package. KAN additionally requires PyTorch and pyKAN; MLP requires CUDA-enabled PyTorch; LR requires scikit-learn. The study-specific notebook dependencies are not declared in Poetry configuration, so `poetry install` alone does not guarantee a working notebook environment. Select an environment that already provides the relevant dependencies; the notebooks do not install packages.

## Usage

Open any of [`notebooks/kan-poc.ipynb`](notebooks/kan-poc.ipynb), [`notebooks/mlp-poc.ipynb`](notebooks/mlp-poc.ipynb), or [`notebooks/lr-poc.ipynb`](notebooks/lr-poc.ipynb) in Jupyter with the dataset and dependencies available. Their full-suffix evaluation and CPU timing are validation studies; the final chronological test remains reserved. The MLP notebook requires PyTorch with CUDA support and threadpoolctl in addition to the common notebook environment; training requires CUDA, then the fitted model is moved to CPU for all inference and codec evaluation. The LR notebook uses scikit-learn 1.7 for its study-specific fit and optionally uses threadpoolctl for controlled timing; timing is not reported when threadpoolctl is unavailable. See [`notebooks/lr-poc.ipynb`](notebooks/lr-poc.ipynb) for the LR study. A previously disclosed test-prefix check is historical provenance only and is not produced by the current run.

## License

MIT License; see `LICENSE`.
