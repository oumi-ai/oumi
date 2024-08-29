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

Reach out to <matthew@learning-machines.ai> if you have problems with access.

## User Setup

To install LeMa, you can use pip:
`pip install 'lema[cloud,dev,train]'`

## Troubleshooting

1. Pre-commit hook errors with vscode
   - When committing changes, you may encounter an error with pre-commit hooks related to missing imports.
   - To fix this, make sure to start your vscode instance after activating your conda environment.

     ```shell
     conda activate lema
     code .  # inside the LeMa directory
     ```
