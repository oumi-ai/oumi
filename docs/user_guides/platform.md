# Oumi Enterprise Platform

```{admonition} Experimental
:class: warning
Oumi Enterprise platform integration is under active development. APIs and
URI shapes may change.
```

The Oumi Enterprise platform is a hosted service (separate from this OSS
package) that stores datasets, models, judges/evaluators, and recipes for
your team, runs training/eval/synthesis jobs on managed compute, and serves
fine-tuned models behind an OpenAI-compatible inference endpoint.

This guide covers how to use enterprise-managed resources **from your local
oumi CLI and Python code**: reference platform datasets and models in any
config, run jobs locally or remotely, and push results back to the platform.

```{admonition} Related
:class: note
- To *launch training* on raw cloud clusters (AWS / GCP / SkyPilot), see
  {doc}`/user_guides/launch/launch`.
- To *deploy* a trained model to Fireworks or Parasail, see
  {doc}`/user_guides/deploy`.
- The {doc}`/user_guides/mcp` server also exposes platform resources to
  MCP clients like Claude Desktop and Cursor.
```

## When to Use It

Reach for the platform integration when:

- You want a single place where your team's datasets, judges, and trained
  models live, with versioning.
- You want training/eval/synthesis runs to land somewhere observable
  (web UI, billing, logs) instead of vanishing into a laptop's `output/`.
- You want to mix-and-match local development with hosted compute — for
  example, iterate on a config locally and then re-run the same command
  with `--remote` on the platform.

If none of those apply, the rest of the OSS CLI is unchanged and works
exactly as documented elsewhere.

## Logging In

Save your platform credentials once:

```bash
oumi platform login
# API key (hidden):  ****
# API URL [https://api.oumi.ai]:
```

This writes `~/.config/oumi/credentials.json` (mode `0600`). Subsequent
commands resolve credentials in this order:

1. Explicit arguments (e.g. `--project p-7` on a command, or
   `Client(api_key=...)` in Python).
2. Environment variables: `OUMI_API_URL`, `OUMI_API_KEY`, `OUMI_PROJECT_ID`.
3. The credentials file.

Move credentials to a different path with `OUMI_CREDENTIALS_FILE`. Honor
`XDG_CONFIG_HOME` if you set it.

Useful related commands:

```bash
oumi platform whoami    # confirm which URL/project the current key targets
oumi platform logout    # delete the on-disk credentials file
```

## The `oumi://` URI Scheme

Any oumi config field that takes a name or path can take an `oumi://` URI
instead. The first time the field is read, the resource is fetched from the
platform and cached under `~/.cache/oumi/platform/`.

```
oumi://<kind>/<resource_id>[@<version>]

kind ∈ { datasets, models, judges, evaluators, recipes }
```

Behaviour by kind:

| Scheme                                  | Resolves to                                          | Cached?                |
|----------------------------------------|------------------------------------------------------|------------------------|
| `oumi://datasets/<id>[@v]`             | local path to a `.jsonl` file                        | yes, per `(id, version)` |
| `oumi://models/<id>[@v]`               | local directory containing model artifacts          | yes, per `(id, version)` |
| `oumi://judges/<id>[@v]`               | evaluator dict (validated as `evaluatorType==judge`) | no (cheap)             |
| `oumi://evaluators/<id>[@v]`           | evaluator dict (any type)                            | no                     |
| `oumi://recipes/<id>[@v]`              | recipe dict (TrainingConfig-shaped)                  | no                     |

If the version is omitted, `oumi` calls the platform to find the current
version and uses that as the cache key — so `oumi://datasets/foo` stays
correct as new versions land on the platform.

Where you'd use them:

```yaml
# A training config that pulls dataset and base model from the platform.
model:
  model_name: oumi://models/my-llama3-sft
data:
  train:
    datasets:
      - dataset_name: oumi://datasets/customer-support@v4
```

```bash
# A judge run that pulls a hosted judge:
oumi judge dataset \
  --config oumi://judges/quality-suite \
  --input conversations.jsonl
```

You can also use the resolver directly from Python:

```python
from oumi.platform import resolve

path = resolve("oumi://datasets/customer-support@v4")
# -> PosixPath('/Users/me/.cache/oumi/platform/datasets/proj/customer-support/v4/data.jsonl')
```

Override the cache root with `OUMI_CACHE_DIR=…` (entries land in
`$OUMI_CACHE_DIR/platform`).

## Running Jobs on the Platform

You can run training, evaluation, and synthesis on managed platform compute
by adding `--remote` to the existing commands. The local config is the
source of truth: it's parsed and validated locally, then submitted to the
platform's job-submission endpoint.

```bash
# Train on platform compute, wait for it to finish.
oumi train \
  --config recipes/sft_qwen3.yaml \
  --data.train.datasets[0].dataset_name oumi://datasets/customer-support \
  --remote --wait

# Same, but detached: returns immediately with an operation id.
op=$(oumi train --config recipes/sft_qwen3.yaml --remote --detach)
oumi platform operations status "$op" --wait

# Evaluation:
oumi evaluate \
  --config eval.yaml \
  --model.model_name oumi://models/sft-qwen3-experiment-7 \
  --remote --wait

# Synthesis: write straight back to a new platform dataset.
oumi synth \
  --config synth.yaml \
  --output_path oumi://datasets/q3-curated-data \
  --remote --wait
```

Available remote flags on `oumi train`, `oumi evaluate`, and `oumi synth`:

| Flag             | Default      | Meaning                                                       |
|------------------|--------------|---------------------------------------------------------------|
| `--remote`       | unset        | Submit to the platform instead of running locally.            |
| `--wait` / `--detach` | `--wait` | Block until the platform operation reaches a terminal state. |
| `--project`      | from creds   | Override the platform project id for this submission.        |

```{admonition} Server prerequisite
:class: warning
`--remote` calls `POST /v1/projects/{project_id}/jobs:submit` on the
platform. If the server hasn't shipped that route yet, the command exits
with an actionable error pointing at the platform upgrade — local runs
are unaffected.
```

If you prefer the launcher abstraction (same end result), use it directly:

```bash
oumi launch up --config job.yaml --cluster oumi-platform
```

`job.yaml` is a `JobConfig` whose `run` line contains the underlying oumi
command. The launcher cloud `oumi-platform` infers the job kind
(`train`/`evaluate`/`synth`/`judge`/`infer`) from the run command, or you
can force it via `envs.OUMI_JOB_KIND`.

## Inference Against Hosted Models

Set the engine type to `OUMI_PLATFORM` and any oumi inference path
(`oumi infer`, the judge runner, evaluators, your own `SimpleJudge`) talks
to the platform's OpenAI-compatible endpoint:

```bash
oumi infer \
  --engine OUMI_PLATFORM \
  --model oumi://models/sft-qwen3-experiment-7@latest \
  --interactive
```

In YAML:

```yaml
engine: OUMI_PLATFORM
model:
  model_name: oumi://models/sft-qwen3-experiment-7@latest
```

The engine reads `OUMI_API_URL` (falling back to the credentials file, then
the public default) and `OUMI_API_KEY`. No further wiring is needed.

## `oumi platform` CLI

A small set of commands for everyday platform operations without leaving
your terminal:

| Command                                          | What it does                                            |
|--------------------------------------------------|---------------------------------------------------------|
| `oumi platform login` / `logout` / `whoami`      | Manage and inspect saved credentials.                   |
| `oumi platform datasets list`                    | List datasets in your project.                          |
| `oumi platform datasets pull <id> [--out <p>]`   | Download a dataset to a local file.                     |
| `oumi platform datasets push <path> [--name]`    | Upload a local file as a new platform dataset.          |
| `oumi platform models list`                      | List models in your project.                            |
| `oumi platform models pull <id> [--version]`     | Download a model checkpoint directory.                  |
| `oumi platform judges list`                      | List judges (evaluators of type `judge`).               |
| `oumi platform operations status <id> [--wait]`  | Show the status of a long-running platform operation.   |
| `oumi platform operations stop <id>`             | Request cancellation of an operation.                   |

Every command accepts `--project <id>` to override the default project.

## Python API

For programmatic use:

```python
from oumi.platform import Client, get_default_client, resolve

# Process-wide singleton client; reads credentials from env/file once.
client = get_default_client()

# Resource access:
for ds in client.datasets.list()["datasets"]:
    print(ds["displayName"])

dataset = client.datasets.get("123")
client.datasets.download("123", "./local.jsonl")
client.datasets.upload("./local.jsonl", display_name="my-data")

# Operation polling:
op = client.operations.wait(operation_id, timeout=600)

# URI resolution:
path = resolve("oumi://datasets/my-dataset")
```

Error handling:

```python
from oumi.platform import (
    CredentialsNotFoundError,
    PlatformAPIError,
    PlatformAuthError,
    PlatformOperationError,
)
```

`PlatformAuthError` maps to 401/403, `PlatformAPIError` carries
`status_code` and `response_body` for any other non-2xx, and
`PlatformOperationError` (raised by `operations.wait`) carries the failing
operation's payload.

## Plugin Extension Point

Third-party packages can register additional datasets, judges, clouds, or
analyzers with the OSS registry without touching `oumi` itself. Declare an
entry point under the `oumi.plugins` group:

```toml
# pyproject.toml of your plugin package
[project.entry-points."oumi.plugins"]
my_company = "my_company.oumi_plugin"
```

The first time any code reads from `oumi.core.registry.REGISTRY`, the
referenced module is imported, which is where you call
`register_dataset(...)` / `register_cloud_builder(...)` / etc. Broken
plugins log a warning and are skipped rather than blocking the rest of
oumi from booting.

## MCP Tools

When the `oumi-mcp` server (see {doc}`/user_guides/mcp`) starts with
`OUMI_API_KEY` set in its environment, it exposes six additional tools to
the connected MCP client:

| Tool                       | Returns                                |
|----------------------------|----------------------------------------|
| `list_platform_datasets`   | Datasets in your project.              |
| `get_platform_dataset`     | A single dataset payload.              |
| `list_platform_models`     | Models in your project.                |
| `get_platform_model`       | A single model payload.                |
| `list_platform_judges`     | Judges in your project.                |
| `get_platform_operation`   | An operation's current status.         |

Each returns `{"ok": true, "result": ...}` on success and
`{"ok": false, "error": "..."}` on failure (including "not logged in"), so
the MCP server stays usable even when the user only operates locally.

## Limitations

Things known not to work yet, plus where to track them:

- `oumi judge --remote` — judge submissions require a separate input-file
  upload step that isn't wired through yet.
- `oumi train` with `--training.output_dir oumi://models/...` — the model
  import API has a different shape than the dataset upload flow and is
  pending design.
- Datasets over 5 GB — multipart upload sessions return an explicit
  not-yet-supported error today.
- Launcher log streaming through `oumi-platform` cloud — currently raises;
  use the platform web UI or `oumi platform operations status` instead.
