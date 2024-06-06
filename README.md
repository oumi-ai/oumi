# Learning Machines (LeMa)

Learning Machines modeling platform

## Description

lema is a learning machines modeling platform that allows you to build and train machine learning models easily.

- Easy-to-use interface for data preprocessing, model training, and evaluation.
- Support for various machine learning algorithms and techniques.
- Visualization tools for model analysis and interpretation.
- Integration with popular libraries and frameworks.

## Features

- [x] Easily run in a locally, jupyter notebook, vscvode debugger, or remote cluster
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

   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Then follow "Next steps" (shown after installation) to add `brew` into `.zprofile`

   ```
   brew install gh
   ```

#### 3.2 Instructions for **Linux**, including [WSL](https://learn.microsoft.com/en-us/windows/wsl/)

  Follow https://github.com/cli/cli?tab=readme-ov-file#conda

   ```
   conda install gh --channel conda-forge
   ```

### 4. Authorize Github CLI (easier when using SSH protocol)

   ```
   gh auth login
   ```

### 5. Set your Github name and email

   ```
   git config --global user.name "YOUR_NAME"
   git config --global user.email YOUR_USERNAME@learning-machines.ai

   ```

### 6. Clone the lema repository

   ```
   gh repo clone openlema/lema
   ```

### 7. Install lema package and its dependencies

   ```
   cd lema
   pip install -e '.[all]'
   ```

### 8. Install pre-commit hooks

   ```
   pre-commit install
   ```

### 9. [optional] Add a lema shortcut in your environment {.zshrc or .bashrc}

    ```
    alias lema="cd ~/<YOUR_PATH>/lema && conda activate lema"
    ```

    Ensure that this works with:
    ```
    source ~/{.zshrc or .bashrc}
    lema
    ```

### 10. [optional] Install [Git Credential Manager](https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls) for authentication management.

## User Setup

To install lema, you can use pip:
`pip install lema[cloud,dev,train]`


## Training on a cloud cluster
To train on a cloud GPU cluster, first make sure to have all the dependencies installed:
```python
pip install lema[cloud]
```

Then setup your cloud credentials:
- [Runpod](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#runpod-cloud)
- [Lambda Labs](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#lambda-cloud)

Your environment should be ready! Use this to check:
```python
sky check
```

You can look at the existing clusters with the following command:
```python
sky status
```

To see the available GPUs, you can use the following command:
```python
sky show-gpus
```

To launch a job on the cloud, you can use the following command:
```python
# edit the configs/skypilot/sky.yaml file to your needs
sky launch -c lema-cluster configs/skypilot/sky.yaml
```

Remember to stop the cluster when you are done to avoid extra charges. You can either do it manually, or use the following to automatically take it down after 10 minutes of inactivity:
```python
sky autostop lema-cluster -i 10
```
