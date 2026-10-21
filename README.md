# Real-Time Predictive Coding Compression Evaluation

**Authors:** Athulya Weerakoon, Arosh Upathilak ([GitHub](https://github.com/Arosh-Upathilak))  
**Affiliation:** University of Jaffna, Sri Lanka  
**License:** MIT License (see `LICENSE` file)

## Overview

This repository contains predictive-coding compression tools and research notebooks. The active proof-of-concept notebook is [`notebooks/kan-poc.ipynb`](notebooks/kan-poc.ipynb): it uses a fixed KAN predictor, signed residual ticks, and ANS coding to study the Air Temperature series. It is a fixed proof of concept, not an ideal optimized or deployment model. The notebook imports reusable study helpers from `compression_evaluation.poc` (and plotting helpers from `compression_evaluation.poc_plots`).

The six earlier root notebooks are preserved byte-for-byte under [`notebooks/old_notebooks/`](notebooks/old_notebooks/): `bpc-with-ans.ipynb`, `lpc-with-ans.ipynb`, `mtkan-with-ans-2-precisions copy.ipynb`, `mtkan-with-ans.ipynb`, `neural-network.ipynb`, and `tans.ipynb`. They are archived, not active entry points. Their original relative dataset/import paths may no longer resolve from the archive directory; adjust paths if running them. The two previously archived notebooks remain there unchanged. Checkpoints and image files are not part of this notebook reorganization.

## Dataset and environment

Datasets are not included. The active notebook expects `datasets/AriviyalN_Data_Air_Temperature.csv` at the repository root. It requires Python with NumPy, pandas, matplotlib, PyTorch, pykan, and this local package; its managed GPU environment also uses CUDA-enabled PyTorch. The study-specific notebook dependencies are not declared in Poetry configuration, so `poetry install` alone does not guarantee that the notebook environment is available. Select an environment that already provides these dependencies; the notebook does not install packages. No model, data, or production migration is performed by importing the toolkit.

## Usage

Open `notebooks/kan-poc.ipynb` in Jupyter and run it with the dataset and dependencies available. Its full-suffix evaluation and optional symbolic/CPU timing are validation studies; the final chronological test remains reserved. A previously disclosed test-prefix check is historical provenance only and is not produced by the current run.

## License

MIT License; see `LICENSE`.
