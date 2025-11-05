
Causal inference code for GNN-TARNET models and baseline models for comparison.
Code for the paper "Graph Neural Networks for Individual Treatment Effect Estimation".

Run ./docker_command.sh to build the Docker image and install dependencies (ensure Docker is installed and running). Inside the container, run the main experiment driver to train or evaluate models.

Compact repository for training and evaluating graph neural network models for individual treatment effect estimation, including the GNN-TARNET architecture used in the associated paper. The repository supports datasets with graph-structured data (e.g., social networks or molecular graphs) for use cases such as personalized medicine or recommendation systems.

## Table of contents
- Overview
- Requirements
- Quick start
- Running experiments
- Citation
- License

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

## Quick start
1. Clone the repository.
2. Use the included Docker helper script to build and launch a container with all dependencies.
3. Inside the container, run the main experiment driver to train or evaluate models (see the Configuration section or the script's CLI help for available options).

## Running experiments
- Training and evaluation are controlled by the primary experiment script. Check the script's command-line help for configurable parameters such as dataset selection, model type, learning rate, batch size, and logging directory.
- Hyperparameter searches and reproducible seed control are supported via configuration files or CLI switches.

## Citation
If you use this code in your research, please cite the related publication describing the GNN-TARNET approach:

A. Sirazitdinov, M. Buchwald, V. Heuveline and J. Hesser, "Graph Neural Networks for Individual Treatment Effect Estimation," in IEEE Access, vol. 12, pp. 106884-106894, 2024, doi: 10.1109/ACCESS.2024.3437665.

## License
See the LICENSE file for license details.

For questions about usage or to report problems, open an issue in this repository.
