# Reproducibility Study of *Transformers Can Do Bayesian Inference*

This repository includes code for a reproducibility study of **“Transformers Can Do Bayesian Inference”**. The project tests whether Prior-Data Fitted Networks (PFNs), implemented with Transformer-based models, can approximate Bayesian inference from synthetic prior-sampled datasets, including Gaussian processes with fixed hyperparameters and hyperpriors. We also extend the PFN framework to desgining optimal strategies for complex quantum systems, where PFN surrogates are trained on synthetic quantum-information datasets to optimise a control channel for decoupling a quantum system from its environment.

## Repository Structure

```text
Quantum_ML/
├── datasets/              # Dataset generation scripts
├── model/                 # Transformer, regressor, distribution, sampling and training code
├── utility/               # Quantum/channel-related utility functions
├── main.ipynb             # Main experiment notebook
├── main_unitarity3d.ipynb # Additional experiment notebook
├── train_unitarity3d.py   # Training script
├── pyproject.toml         # Project configuration
└── README.md              # Project notes
```

## Setup

Create a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies based on the project configuration:

```bash
pip install -e .
```

## Usage

To run the main experiments, open the notebooks:

```bash
jupyter notebook main.ipynb
```

or open the notbooks:

```bash
main_unitarity3d.ipynb
```
