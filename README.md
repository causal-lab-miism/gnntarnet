
# GNN-TARNET

Causal inference code for GNN-TARNET models and baseline models for comparison.

Code for the paper "Graph Neural Networks for Individual Treatment Effect Estimation".

This repository provides a compact framework for training and evaluating graph neural network models for individual treatment effect estimation, including the GNN-TARNET architecture used in the associated paper. The repository supports datasets with graph-structured data (e.g., social networks or molecular graphs) for use cases such as personalized medicine or recommendation systems.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Citation](#citation)
- [License](#license)

## Overview
This project provides code to train, validate, and compare GNN-based and baseline models for individualized treatment effect estimation on graph-structured data.

## Features
- Implementations of GNN-TARNET and baseline models
- Training, evaluation, and hyperparameter sweep utilities
- Dockerized environment for reproducible runs
- Example configuration and dataset loaders

## Requirements
- Docker (recommended for reproducibility)
- Python 3.8+ (if running without Docker)
- CUDA and a recent NVIDIA driver for GPU training (optional)

## Installation

### Using Docker (Recommended)

1. Ensure Docker is installed and running on your system.
2. Build the Docker image and run a container:
   ```bash
   ./docker_commands.sh
   ```
   This will build the image and start a container with all dependencies installed.

### Without Docker

If you prefer to run without Docker, ensure you have Python 3.8+ installed, then install the required dependencies:
```bash
pip install tensorflow tensorflow-probability==0.19.0 seaborn codecarbon==2.3.1 causal-learn keras-tuner==1.1.3 tf2onnx==1.14.0 onnxruntime==1.15.1 pandas numpy scikit-learn matplotlib
```

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/causal-lab-miism/gnntarnet.git
   cd gnntarnet
   ```

2. Build and launch the Docker container (or install dependencies as shown above):
   ```bash
   ./docker_commands.sh
   ```

3. Inside the container, run the main experiment driver:
   ```bash
   python main_hyper.py --model-name GNNTARnet --dataset-name ihdp_a
   ```

## Running Experiments

Training and evaluation are controlled by the `main_hyper.py` script. The following command-line options are available:

### Available Models
- `GNNTARnet`: Graph Neural Network TARNET (default)
- `TARnet`: Treatment-Agnostic Representation Network
- `CFRNet`: Counterfactual Regression Network
- `SLearner`: S-Learner baseline
- `TLearner`: T-Learner baseline
- `GANITE`: Generative Adversarial Nets for Inference of Individualized Treatment Effects
- `TEDVAE`: Treatment Effect Disentangled Variational Autoencoder

### Available Datasets
- `ihdp_a`: IHDP dataset variant A
- `ihdp_b`: IHDP dataset variant B
- `jobs`: Jobs dataset

### Example Commands

Train the GNN-TARNET model on IHDP dataset variant A:
```bash
python main_hyper.py --model-name GNNTARnet --dataset-name ihdp_a
```

Train a baseline T-Learner model on the JOBS dataset:
```bash
python main_hyper.py --model-name TLearner --dataset-name jobs
```

Run with custom hyperparameters:
```bash
python main_hyper.py --model-name GNNTARnet --dataset-name ihdp_a --num 100 --num_layers 5
```

### Command-Line Options
- `--model-name`: Model architecture to use (default: `GNNTARnet`)
- `--dataset-name`: Dataset to train on (default: `ihdp_a`)
- `--tuner-name`: Hyperparameter tuner type (default: `random`)
- `--num`: Number of experimental runs (default: 1)
- `--num_layers`: Number of GNN layers (default: 5)
- `--defaults`: Use default hyperparameters (default: `True`)

For a complete list of options, run:
```bash
python main_hyper.py --help
```

## Citation
If you use this code in your research, please cite the related publication describing the GNN-TARNET approach:

A. Sirazitdinov, M. Buchwald, V. Heuveline and J. Hesser, "Graph Neural Networks for Individual Treatment Effect Estimation," in IEEE Access, vol. 12, pp. 106884-106894, 2024, doi: 10.1109/ACCESS.2024.3437665.

## License
See the LICENSE file for license details.

For questions about usage or to report problems, open an issue in this repository.
