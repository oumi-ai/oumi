# Learning Machines (LeMa)

Learning Machines modeling platform

## Description

lema is a learning machines modeling platform that allows you to build and train machine learning models easily.

- Easy-to-use interface for data preprocessing, model training, and evaluation.
- Support for various machine learning algorithms and techniques.
- Visualization tools for model analysis and interpretation.
- Integration with popular libraries and frameworks.

## [WIP] Features

- [ ] Easily run in a locally, jupyter notebook, vscvode debugger, or remote cluster
- [ ] Full finetuning using SFT, DPO

## [TODO] Open Design Questions
- [ ] What is a good data abstraction of instruction finetuning datasets ?

## Dev Environment Setup


1. Install homebrew (the command below was copied from brew.sh)

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

4. Clone the lema repository

   ```
   gh repo clone openlema/lema
   ```

5. Install Miniconda

   https://docs.anaconda.com/free/miniconda/miniconda-install/

[comment]: <> (This is a package/environment manager that we mainly need to pull all the relevant python packages via pip)


6. Create a new environment for lema and activate it

   ```
   conda create -n leva python=3.11
   activate lema environment
   ```

7. Install lema:

   ```
   cd lema
   pip install -e .
   ```

8. Install pre-commit hooks

   ```
   pip install pre-commit
   pre-commit install
   ```

9. [optional] Add a lema shortcut in your environment {.zshrc or .bashrc}

    ```
    alias lema="cd ~/<YOUR_PATH>/lema && conda activate lema"
    ```

    Ensure that this works with:
    ```
    source ~/{.zshrc or .bashrc}
    lema
    ```

## [WIP] User Setup

To install lema, you can use pip:
`pip install lema[dev,train]`
