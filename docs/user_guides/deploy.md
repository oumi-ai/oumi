# Deploying Models

Oumi provides a top-level `oumi deploy` command for taking a trained or downloaded model and standing it up as a managed inference endpoint on a third-party provider. Today it supports **Fireworks AI** and **Parasail.io**.

```{admonition} Related
:class: note
- To *launch training* on remote clusters, see {doc}`/user_guides/launch/launch`.
- To *call* a deployed endpoint, see {doc}`/user_guides/infer/inference_engines`.
```

## Overview

The deploy workflow has three stages, each exposed as a sub-command:

1. **Upload** — push the model (full weights or a LoRA adapter) to the provider.
2. **Create endpoint** — provision hardware and start serving the uploaded model.
3. **Test / use** — smoke-test the endpoint and then call it with any inference engine.

For the common case, `oumi deploy up` runs all three stages end-to-end from a single YAML config.

## Prerequisites

- A provider account and API key exported in your shell:
  - Fireworks: `FIREWORKS_API_KEY`
  - Parasail:  `PARASAIL_API_KEY`
- For Fireworks, the model must exist on your local disk (HuggingFace download or an Oumi training output).

## Quick Start: End-to-End Deploy

```bash
oumi deploy up --config configs/examples/deploy/fireworks_deploy.yaml
```

The `--config` YAML matches the {py:class}`~oumi.deploy.deploy_config.DeploymentConfig` schema:

```yaml
# configs/examples/deploy/fireworks_deploy.yaml
model_source: /path/to/my-finetuned-model/   # local directory
provider: fireworks                           # fireworks | parasail
model_name: my-finetuned-model-v1             # display name on the provider
model_type: full                              # full | adapter
# base_model: accounts/fireworks/models/llama-v3p1-8b-instruct  # required if adapter

hardware:
  accelerator: nvidia_h100_80gb               # see `oumi deploy list-hardware`
  count: 2

autoscaling:
  min_replicas: 1
  max_replicas: 4

test_prompts:
  - "Hello, how are you?"
```

Any of `model_source`, `provider`, and `hardware` can be overridden on the CLI, e.g.:

```bash
oumi deploy up \
  --config fireworks_deploy.yaml \
  --model-path /tmp/llama3-8b \
  --hardware nvidia_a100_80gb
```

`oumi deploy up` will upload the model, wait for it to be ready, create an endpoint, optionally run any `test_prompts`, and print the endpoint URL.

## Sub-Commands

| Command                         | What it does                                                         |
|---------------------------------|----------------------------------------------------------------------|
| `oumi deploy up`                | Full pipeline: upload → create endpoint → test                        |
| `oumi deploy upload`            | Upload a model only                                                   |
| `oumi deploy create-endpoint`   | Create an endpoint for a previously uploaded model                    |
| `oumi deploy list`              | List all deployments on the provider                                  |
| `oumi deploy list-models`       | List uploaded models                                                  |
| `oumi deploy list-hardware`     | List hardware options available for a provider                        |
| `oumi deploy status`            | Show endpoint state, replica counts, URL                              |
| `oumi deploy start` / `stop`    | Start or stop an existing endpoint (pause to save cost)               |
| `oumi deploy delete`            | Delete an endpoint                                                    |
| `oumi deploy delete-model`      | Delete an uploaded model                                              |
| `oumi deploy test`              | Send a sample request to an endpoint                                  |

Add `--help` to any sub-command for the exact flags it accepts, or see {doc}`/cli/commands`.

## Using a Deployed Endpoint

Once `oumi deploy up` reports `RUNNING`, point any Oumi inference engine at the returned URL. For Fireworks:

```python
from oumi.inference import FireworksInferenceEngine
from oumi.core.configs import ModelParams

engine = FireworksInferenceEngine(
    model_params=ModelParams(model_name="my-finetuned-model-v1")
)
```

For Parasail:

```python
from oumi.inference import ParasailInferenceEngine
from oumi.core.configs import ModelParams

engine = ParasailInferenceEngine(
    model_params=ModelParams(model_name="my-finetuned-model-v1")
)
```

Both engines are documented in {doc}`/user_guides/infer/inference_engines`.

## Tips

- **Cost control.** Use `oumi deploy stop <endpoint>` to pause an endpoint without deleting it; `start` brings it back online. Set `autoscaling.min_replicas: 0` if the provider supports scale-to-zero.
- **LoRA adapters.** Set `model_type: adapter` and a matching `base_model` to deploy a LoRA adapter on top of a hosted base model. This is usually cheaper than a full model.
- **Smoke tests.** `test_prompts` at the bottom of the YAML run automatically after `oumi deploy up` finishes — quick sanity check before sending real traffic.

## See Also

- {doc}`/user_guides/infer/inference_engines` — calling the deployed endpoint
- {doc}`/user_guides/launch/launch` — launching training jobs on remote clusters
- {doc}`/cli/commands` — CLI reference
