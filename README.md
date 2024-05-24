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


1. Install homebrew (the command below was copied from www.brew.sh)

   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

   Then follow "Next steps" (shown after installation) to add `brew` into `.zprofile`

2. Install GitHub CLI

   ```
   brew install gh
   ```

3. Authorize Github CLI (easier when using SSH protocol)

   ```
   gh auth login
   ```

4. Set your Github name and email

   ```
   git config --global user.name "YOUR_NAME"
   git config --global user.email YOUR_USERNAME@openlema.com

   ```

5. Clone the lema repository

   ```
   gh repo clone openlema/lema
   ```

6. Install Miniconda

   https://docs.anaconda.com/free/miniconda/miniconda-install/

[comment]: <> (This is a package/environment manager that we mainly need to pull all the relevant python packages via pip)


7. Create a new environment for lema and activate it

   ```
   conda create -n lema python=3.11
   conda activate lema
   ```

8. Install lema package and its dependencies

   ```
   cd lema
   pip install -e .
   ```

9. Install pre-commit hooks

   ```
   pip install pre-commit
   pre-commit install
   ```

10. [optional] Add a lema shortcut in your environment {.zshrc or .bashrc}

    ```
    alias lema="cd ~/<YOUR_PATH>/lema && conda activate lema"
    ```

    Ensure that this works with:
    ```
    source ~/{.zshrc or .bashrc}
    lema
    ```

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

You environement should be read! Use this to check
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

Remember to stop the cluster when you are done to avoid extra charges. You can either do it manuall, or use the following to automatically take it down after 10 minutes of inactivity:
```python
sky autostop lema-cluster -i 10
```
