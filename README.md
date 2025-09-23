# Custom_Encoder

This is the code for the paper "A Large-scale Comparison of Customized Feature Encodings under Semi-supervised Learning"

Research on custom encoders is becoming increasingly popular. However, traditionally, defining a custom encoder often required modifying the overall model structure, which can be cumbersome. 
Our framework allows flexible encoder integration without modifying the model, and supports any PyTorch architecture.
Additionally, we provide a set of APIs and a pre-configured dataset (as used in our experiments) to help users quickly evaluate encoder performance.

# Dependency

- python>=3.8
- pandas>=2.0.0
- numpy==1.24.4
- pyyaml==5.3.1
- ucimlrepo>=0.07
- scikit-learn>=1.5.0
- munkres>=1.1.4
- openml>=0.14.2

# Usage

There are two ways to install this project.
Use `pip` if you want to integrate it with any model.
Use `git clone` if you want to reproduce or compare experiments.

## Installation

* PyPI (pip):

```console
$ pip install CE-Module
```

* Clone the repo:

```console
$ git clone https://github.com/DEPhantom/Custom_Encoder.git
```

## Quick Start

## Configuration

There is a configuration file "config/config.yaml", where one can edit both the training and test options.

## Citation
Please cite our work if you find the dataset or the code useful in your work.
```
...
```
