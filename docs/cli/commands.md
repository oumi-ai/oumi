# CLI Reference

This page contains a complete reference of all CLI commands available in Oumi.

For detailed guides and examples of specific areas (training, inference, evaluation, etc.), please refer to the corresponding user guides in the documentation.

## CLI Overrides

Any Oumi command which takes a config path as an argument (`train`, `evaluate`, `infer`, etc.) can override parameters from the command line. For example:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.max_steps 20 \
  --training.learning_rate 1e-4 \
  --data.train.datasets[0].shuffle true \
  --training.output_dir output/smollm-135m-sft
```

Oumi uses [OmegaConf](https://omegaconf.readthedocs.io/) to parse the configs from YAML files, and to parse the command line overrides. OmegaConf allows a Pythonic specification of parameters to override with dot-separated syntax, as seen above. Note that for lists (ex. `data.train.datasets`), you can specify the index either with brackets (`[0]`) or dot notation (`.0`).

With OmegaConf, you can set the value of an entire dictionary or list, in addition to overriding individual primitive values. For example:

```bash
# Override one entry in the list. Note that the new dict is merged with the existing
# one, so the existing value of `"dataset_name": "yahma/alpaca-cleaned"` is kept.
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --data.train.datasets[0] '{"shuffle": True, "sample_count": 100}'

# Override the list, in this case to add a new entry.
# Note that we redundantly specify an existing entry in the list here.
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --data.train.datasets '[{"dataset_name": "yahma/alpaca-cleaned"}, {"dataset_name": "CohereForAI/aya_dataset"}]'
```

```{warning}
OmegaConf doesn't readily support adding/deleting entries in a list from command line overrides using index notation. Instead, you need to set the value of the entire list, or modify the YAML config.
```

```{warning}
By default, when you override a dict value with another dict, the items from both will be merged, preferring the value from the overriding dict in case of an existing key. This is equivalent to Python dict merging behavior, ex. `new_dict = {**old_dict, **override_dict}`
```

## Training

For a detailed guide on training, see {doc}`/user_guides/train/train`.

```{typer} oumi.cli.main.app.train
  :prog: oumi train
  :make-sections:
  :preferred: svg
  :theme: monokai
  :width: 80
```

## Evaluation

For a detailed guide on evaluation, see {doc}`/user_guides/evaluate/evaluate`.

```{typer} oumi.cli.main.app.evaluate
  :prog: oumi evaluate
  :make-sections:
  :preferred: svg
  :theme: monokai
  :width: 80
```

## Inference

For a detailed guide on inference, see {doc}`/user_guides/infer/infer`.

```{typer} oumi.cli.main.app.infer
  :prog: oumi infer
  :make-sections:
  :preferred: svg
  :theme: monokai
  :width: 80
```

## Judge

For a detailed guide on judging, see {doc}`/user_guides/judge/judge`.

```{typer} oumi.cli.main.app.judge
  :prog: oumi judge
  :make-sections:
  :show-nested:
  :preferred: svg
  :theme: monokai
  :width: 80
```

## Launch

For a detailed guide on launching jobs, see {doc}`/user_guides/launch/launch`.

```{typer} oumi.cli.main.app.launch
  :prog: oumi launch
  :make-sections:
  :show-nested:
  :preferred: svg
  :theme: monokai
  :width: 80
```

## Distributed

For a detailed guide on distributed training, see {doc}`/user_guides/train/train`.

```{typer} oumi.cli.main.app.distributed
  :prog: oumi distributed
  :make-sections:
  :show-nested:
  :preferred: svg
  :theme: monokai
  :width: 80
```

## Environment

This command is a great tool for debugging!

`oumi env` will list relevant details of your environment setup, including python
version, package versions, and Oumi environment variables.

```{typer} oumi.cli.main.app.env
  :prog: oumi env
  :make-sections:
  :preferred: svg
  :theme: monokai
  :width: 80
```
