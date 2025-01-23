# CLI Reference

This page contains a complete reference of all CLI commands available in Oumi.

For detailed guides and examples of specific areas (training, inference, evaluation, etc.), please refer to the corresponding user guides in the documentation.

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
