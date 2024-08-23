# Learning Machines (LeMa)

LeMa is a learning machines modeling platform that allows you to build foundation models end-to-end including data curation/synthesis, pretraining, tuning, and evaluation.

- Easy-to-use interface for data preprocessing, model training, and evaluation.
- Support for various machine learning algorithms and techniques.
- Visualization tools for model analysis and interpretation.
- Integration with popular libraries and frameworks.

## Features

- [x] Easily run in a locally, jupyter notebook, vscode debugger, or remote cluster
- [x] Full finetuning using SFT, DPO

Take a [tour of our repository](https://github.com/openlema/lema/blob/main/notebooks/LeMa%20-%20A%20Tour.ipynb) to learn more!

## Documentation

View our API documentation [here](https://learning-machines.ai/docs/latest/index.html).

Reach out to matthew@learning-machines.ai if you have problems with access.

## Dev Environment Setup

### 1. Install Miniconda

   https://docs.anaconda.com/free/miniconda/miniconda-install/

[comment]: <> (This is a package/environment manager that we mainly need to pull all the relevant python packages via pip)


### 2. Create a new environment for LeMa and activate it

   ```
   conda create -n lema python=3.11
   conda activate lema
   ```

### 3. Install GitHub CLI

#### 3.1 Instructions for Mac

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

### 6. Clone the LeMa repository

   ```shell
   gh repo clone openlema/lema
   ```

### 7. Install LeMa package and its dependencies

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

### 9. [optional] Add a LeMa shortcut in your environment {.zshrc or .bashrc}

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

To install LeMa, you can use pip:
`pip install 'lema[cloud,dev,train]'`


## Training on a cloud cluster
To train on a cloud GPU cluster, first make sure to have all the dependencies installed:
```shell
pip install 'lema[cloud]'
```

Then setup your cloud credentials:
- [Google Cloud](https://github.com/openlema/lema/wiki/Clouds-Setup)
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
You can add the `-a` flag to show all GPUs. Example GPUs include `A100` (40GB), `A100-80GB`, and `A100-80GB-SXM`.

To launch a job on the cloud, you can use the following command:
```shell
sky launch -c lema-cluster configs/skypilot/sky_gpt2.yaml
```
To launch on the cloud of your choice, use the `--cloud` flag, ex. `--cloud gcp`.

Once you have already launched a job, you can use the following command to execute a job on an existing cluster:
```shell
sky exec -c lema-cluster configs/skypilot/sky_gpt2.yaml
```
If you made any code changes to the codebase (not including configs), you need to run
`pip install '.[train]'` in the `run` section of the SkyPilot config to install the
changes on the cluster.

Remember to stop the cluster when you are done to avoid extra charges. You can either do it manually (`sky down lema-cluster`), or use the following to automatically take it down after 10 minutes of inactivity:
```shell
sky autostop lema-cluster -i 10
```

Alternatively, you can include `-i 10` into your `sky launch` command (as shown above).

### Multi-GPU Training on a Single Node

To configure multi-GPU training, edit the `accelerators` section of your SkyPilot config
 to use `N` GPUs. For example, for 2 `A100` GPUs, set `accelerators: {"A100": 2}`.

There are two options for multi-GPU training:
[DDP (Distributed Data Parallel)](https://huggingface.co/docs/transformers/en/perf_train_gpu_many#dataparallel-vs-distributeddataparallel) and
[FSDP (Fully Sharded Data Parallel)](https://huggingface.co/docs/transformers/en/fsdp).
If your model training can run on a single GPU (i.e. one GPU's memory can hold the model,
optimizer state, etc.), then consider using DDP. Otherwise, consider using FSDP, which
shards the model across your GPUs.

#### DDP (Distributed Data Parallel)

To properly configure your machine to do DDP training, either invoke training with the
[`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) command or
[`accelerate launch`](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch#using-accelerate-launch)
 using the `--multi_gpu` flag.

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
     code .  # inside the LeMa directory
     ```
