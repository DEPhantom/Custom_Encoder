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
The configuration is divided into the following four sections:
1. Dataset Options:

| Option Name | Type     | Allowed Values |     Default    | Description                          |
|-------------|----------|----------------|----------------|--------------------------------------|
| `seed`        | int    | Any number     | `42`           | ID of the ucirepo dataset          |
| `workers`     | int    | Any number     | `8`            | Number of workers for DataLoader  |
| `dataset_dir` | string | Any string     | `"./datasets"` | Path to download the datasets        |

2. Training Options:
   
| Option Name      | Type   | Allowed Values                                | Default    | Description                                 |
|------------------|--------|-----------------------------------------------|------------|---------------------------------------------|
| `batch_size`     | int    | Any number                                    | `128`      | Batch size                                  |
| `start_epoch`    | int    | Any number                                    | `0`        | Epoch to start training from                |
| `epochs`         | int    | Any number                                    | `10`       | Total number of training epochs             |
| `finetune_epochs`| int    | Any number                                    | `20`       | Total number of finetuning epochs           |
| `dataset`        | string | `Breast` / `Wine` / `Spambase` / `OpenML`     | `"OpenML"` | Dataset to use                              |
| `dataset_id`     | int    | Any number                                    | `3918`     | ID of the dataset on OpenML                 |
| `dataset_class`  | int    | Any number                                    | `2`        | Number of classes in the OpenML dataset     |


3. Model Architecture Options:
   
| Option Name    | Type    | Allowed Values                                | Default    | Description                                     |
|----------------|---------|-----------------------------------------------|------------|-------------------------------------------------|
| `encoder`      | string  | `None` / `Periodic` / `PieceWise` / `std`     | `"None"`   | Encoding method to use                         |
| `feature_dim`  | int     | Any number                                    | `21`       | Dimension of input features                    |
| `bin_dim`      | int     | Any number                                    | `38`       | Dimension after PieceWise encodin              |
| `emb_dim`      | int     | Any number                                    | `256`      | Dimension of the embedding                     |
| `instance_dim` | int     | Any number                                    | `256`      | Embedding dimension for finetuning             |
| `model_path`   | string  | Any string                                    | `"save"`   | Path to save the trained model                 |
| `reload`       | boolean | `True` / `False`                              | `False`    | Whether to reload an existing model checkpoint |


4. Model Parameters:
   
| Option Name           | Type   | Allowed Values | Default   | Description                     |
|------------------------|--------|----------------|-----------|---------------------------------|
| `learning_rate`        | float  | Any number     | `0.0003`  | Learning rate                   |
| `weight_decay`         | float  | Any number     | `0.0`     | Weight decay                    |
| `instance_temperature` | float  | Any number     | `0.5`     | Temperature for instance loss   |
| `cluster_temperature`  | float  | Any number     | `1.0`     | Temperature for cluster loss    |

Below is an example configuration.
如果要更詳細的教學 請參照...

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
encoder: "None"
feature_dim: 21
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
