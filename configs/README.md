# Configs

This directory contains YAML configs used for training, evaluation, and inference with Oumi, in addition to launching remote jobs for those tasks. We use [Omegaconf](https://omegaconf.readthedocs.io/en/) to load these config files into our [config classes](../src/oumi/core/configs).

As a convention, we name these files according to their corresponding class. For example:

- `*train.yaml`: [TrainingConfig](../src/oumi/core/configs/training_config.py)
- `*eval.yaml`: [EvaluationConfig](../src/oumi/core/configs/evaluation_config.py)
- `*infer.yaml`: [InferenceConfig](../src/oumi/core/configs/inference_config.py)
- `*gcp_job.yaml`: [JobConfig](../src/oumi/core/configs/job_config.py), GCP cloud
- `*polaris_job.yaml`: [JobConfig](../src/oumi/core/configs/job_config.py), Polaris cloud

## Structure

We use the following sub-directories to organize our configs:

- `recipes/`: This directory contains configs for training, evaluation, and inference of common model families. This is a great starting point for most users.
- `projects/`: Configs for fully replicating the training of specific models (ex. Aya).
- `examples/`: Configs for specific use cases. This is less structured than `configs/recipes` and `configs/projects`, and includes more experimental jobs.
