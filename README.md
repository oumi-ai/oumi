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

## [WIP] User Setup

To install lema, you can use pip:
`pip install lema[dev,train]`
