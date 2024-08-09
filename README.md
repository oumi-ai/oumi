# Learning Machines (LeMa)

Learning Machines modeling platform

## Description

lema is a learning machines modeling platform that allows you to build and train machine learning models easily.

- Easy-to-use interface for data preprocessing, model training, and evaluation.
- Support for various machine learning algorithms and techniques.
- Visualization tools for model analysis and interpretation.
- Integration with popular libraries and frameworks.

## Features

- [x] Easily run in a locally, jupyter notebook, vscode debugger, or remote cluster
- [x] Full finetuning using SFT, DPO

## Dev Environment Setup

### 1. Install Miniconda

   https://docs.anaconda.com/free/miniconda/miniconda-install/

[comment]: <> (This is a package/environment manager that we mainly need to pull all the relevant python packages via pip)


### 2. Create a new environment for lema and activate it

   ```
   conda create -n lema python=3.11
   conda activate lema
   ```

### 3. Install GitHub CLI

#### 3.1. Instructions for Mac

   Install Homebrew (the command below was copied from www.brew.sh)

   ```shell
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Then follow "Next steps" (shown after installation) to add `brew` into `.zprofile`

   ```shell
   brew install gh
   ```

#### 3.2 Instructions for **Linux**, including [WSL](https://learn.microsoft.com/en-us/windows/wsl/)

  Follow https://github.com/cli/cli?tab=readme-ov-file#conda

   ```shell
   conda install gh --channel conda-forge
   ```

### 4. Authorize Github CLI (easier when using SSH protocol)

   ```shell
   gh auth login
   ```

### 5. Set your Github name and email

   ```shell
   git config --global user.name "YOUR_NAME"
   git config --global user.email YOUR_USERNAME@learning-machines.ai
   ```

### 6. Clone the lema repository

   ```shell
   gh repo clone openlema/lema
   ```

### 7. Install lema package and its dependencies

   ```shell
   cd lema
   pip install -e '.[all]'
   ```

### 8. Install pre-commit hooks

   ```shell
   pre-commit install  # recommended
   ```

   If you'd like to only run the pre-commits before a push, you can use:
   ```shell
   pre-commit install --install-hooks --hook-type pre-push
   ```

   Alternatively, you can run the pre-commit hooks manually with:
   ```shell
   pre-commit run --all-files
   ```

### 9. [optional] Add a lema shortcut in your environment {.zshrc or .bashrc}

   ```shell
   alias lema="cd ~/<YOUR_PATH>/lema && conda activate lema"
   ```

   Ensure that this works with:

   ```shell
   source ~/{.zshrc or .bashrc}
   lema
   ```

### 10. [optional] Install [Git Credential Manager](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls) for authentication management.

## User Setup

To install lema, you can use pip:
`pip install 'lema[cloud,dev,train]'`


## Training on a cloud cluster
To train on a cloud GPU cluster, first make sure to have all the dependencies installed:
```shell
pip install 'lema[cloud]'
```

Then setup your cloud credentials:
- [Runpod](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#runpod)
- [Lambda Labs](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#lambda-cloud)

Your environment should be ready! Use this to check:
```shell
sky check
```

You can look at the existing clusters with the following command:
```shell
sky status
```

To see the available GPUs, you can use the following command:
```shell
sky show-gpus
```

To launch a job on the cloud, you can use the following command:
```shell
# edit the configs/skypilot/sky.yaml file to your needs
sky launch -c lema-cluster -i 10 configs/skypilot/sky.yaml
```

Remember to stop the cluster when you are done to avoid extra charges. You can either do it manually (`sky down lema-cluster`), or use the following to automatically take it down after 10 minutes of inactivity:
```shell
sky autostop lema-cluster -i 10
```

Alternatively, you can include `-i 10` into your `sky launch` command (as shown above).

### Multi-GPU Training on a single node

If your model fits on 1 GPU, then consider using [DDP (Distributed Data Parallel)](https://huggingface.co/docs/transformers/en/perf_train_gpu_many#dataparallel-vs-distributeddataparallel) with `N` GPUs and data parallelism. Otherwise, consider [FSDP (Fully Sharded Data Parallel)](https://huggingface.co/docs/transformers/en/fsdp).

#### DDP (Distributed Data Parallel)

To start DDP training, edit [configs/skypilot/sky.yaml](configs/skypilot/sky.yaml) and configure it to use `N` GPUs. For example, for two (2) `A40` GPUs:

* Set the `accelerators:` section as follows: `accelerators: {"A40": 2}`

Then run `sky launch ...` as before.

#### FSDP (Fully Sharded Data Parallel)

NOTE: PyTorch FSDP paper: https://arxiv.org/abs/2304.11277

For example, for Phi3 DPO model, there are two related sample configs provided:
* SkyPilot config: [configs/skypilot/sky_fsdp_phi3_dpo.yaml](configs/skypilot/sky_fsdp_phi3_dpo.yaml)
  * Set the `accelerators:` section as follows: `accelerators: {"A40": N}`, where `N` is the number of GPUs to use e.g., `2`.
* [`accelerate`](https://github.com/huggingface/accelerate) config: [configs/accelerate/phi3.fsdp.dpo.yaml](configs/accelerate/phi3.fsdp.dpo.yaml)
  * Set `num_processes: N`, where `N` is the number of GPUs.
  * Update `fsdp_transformer_layer_cls_to_wrap` to match transformer layer class name in your model.
  * Review and tune other parameters in the config, as described in [FSDP Configuration](https://huggingface.co/docs/transformers/main/en/fsdp#fsdp-configuration) and in [accelerate FSDP usage guide](https://huggingface.co/docs/accelerate/en/usage_guides/fsdp). They control various performance trade-offs.

Then run `sky launch ...` as before.


## Troubleshooting
1. Pre-commit hook errors with vscode
   - When committing changes, you may encounter an error with pre-commit hooks related to missing imports.
   - To fix this, make sure to start your vscode instance after activating your conda environment.
     ```shell
     conda activate lema
     code .  # inside the lema directory
     ```
