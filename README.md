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
分為四大類
| Option Name | Type     | Allowed Values         | Default | Description                          |
|-------------|----------|------------------------|---------|--------------------------------------|
| `seed`      | int   | Any number         | `42`     | dataset的id             |
| `workers`   | int   | Any number      | `8`  | DataLoader 的worker數量     |
| `dataset_dir`   | string  | Any string        | `"./datasets"` | datasets的下載路徑      |

| Option Name | Type     | Allowed Values         | Default | Description                          |
|-------------|----------|------------------------|---------|--------------------------------------|
| `batch_size`      | int   | Any number         | `128`     | batch size             |
| `start_epoch`   | int   | Any number      | `0`  | 從哪一個epoch開始訓練     |
| `epochs`   | int  | Any number        | `10` | 訓練的總epoch      |
| `finetune_epochs`     | int   | Any number        | `20` | finetune的總epoch                       |
| `dataset`     | string   | `light` / `dark`        | `"OpenML"` | 使用的dataset                        |
| `dataset_id`      | int   | Any number         | `3918`     | dataset的id             |
| `dataset_class`      | int   | Any number         | `2`     | dataset的分類數量             |

| Option Name | Type     | Allowed Values         | Default | Description                          |
|-------------|----------|------------------------|---------|--------------------------------------|
| `encoder`      | string   | `light` / `dark`       | `"std2"`     | dataset的id             |
| `feature_dim`   | int   | Any number      | `21`  | DataLoader 的worker數量     |
| `encoder_dim`   | int  | Any number        | `0` | datasets的下載路徑      |
| `bin_dim`     | int   | Any number        | `38` | 使用的dataset                        |
| `emb_dim`      | int   | Any number         | `256`     | dataset的id             |
| `instance_dim`   | int   | Any number      | `256`  | DataLoader 的worker數量     |
| `model_path`   | string  | Any string        | `"save"` | datasets的下載路徑      |
| `reload`     | boolean   | `True` / `False`        | `False` | 使用的dataset                        |

| Option Name | Type     | Allowed Values         | Default | Description                          |
|-------------|----------|------------------------|---------|--------------------------------------|
| `learning_rate`      | float   | Any number         | `0.0003`     | dataset的id             |
| `weight_decay`   | float   | Any number      | `0.`  | DataLoader 的worker數量     |
| `instance_temperature`   | float  | Any number        | `0.5` | datasets的下載路徑      |
| `cluster_temperature`     | float   | Any number        | `1.0` | UI theme mode                        |


```sh
# general
seed: 42
workers: 8
dataset_dir: "./datasets"

# train options
batch_size: 128
start_epoch: 0
epochs: 10
finetune_epochs: 20
dataset: "OpenML"

# OpenML dataset options
dataset_id: 3918
dataset_class: 2

# model options
encoder: "std2"
feature_dim: 21
encoder_dim: 0
bin_dim: 38
emb_dim: 256
instance_dim: 256
model_path: "save"
reload: False

# loss options
learning_rate: 0.0003
weight_decay: 0.
instance_temperature: 0.5
cluster_temperature: 1.0

```

## Citation
Please cite our work if you find the dataset or the code useful in your work.
```
...
```
