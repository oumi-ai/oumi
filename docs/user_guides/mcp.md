# MCP Server

```{admonition} Experimental
:class: warning
The Oumi MCP server is under active development. Tools and resources may change.
```

Oumi ships an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that lets an MCP-capable assistant — Claude Desktop, Claude Code, Cursor, and others — discover Oumi's ~500 ready-to-use YAML configs, launch and monitor training / eval / inference jobs (local or cloud), and read built-in workflow guidance, all without leaving the chat.

It's installed as a separate extra and launched as a standalone stdio process. The client spawns it; you never run it by hand.

## When to Use It

The MCP server is a productivity layer on top of the normal `oumi` CLI. Reach for it when you want to:

- **Find a config**: "Find me a config for LoRA fine-tuning Llama 3.1 8B on an A100."
- **Understand a config**: "Explain what this config does and flag anything risky."
- **Launch a job from chat**: "Run `oumi train` on this config locally / on GCP."
- **Babysit a running job**: "Poll the job every minute and tell me when it finishes."
- **Look up API docs**: "How do I configure FSDP2 in Oumi?"

If you'd rather run commands yourself, the CLI is always available and documented in {doc}`/cli/commands`.

## Installation

```bash
pip install "oumi[mcp]"
```

This pulls in `fastmcp`, the `mcp` package, and `httpx`. Two entry points are installed:

```bash
oumi-mcp           # console script — use this one in client configs
python -m oumi.mcp # equivalent, if you prefer module invocation
```

Both start the server on stdio.

```{note}
Path-sensitive tools (job launch, pre-flight, validation) rely on you having a normal Oumi environment wherever the server runs. To launch cloud jobs, also install the per-provider extras (`oumi[aws]`, `oumi[azure]`, `oumi[gcp]`, `oumi[kubernetes]`, `oumi[lambda]`, `oumi[nebius]`, `oumi[runpod]`) and complete `sky check` per {doc}`/user_guides/launch/launch`.
```

## Connecting from an MCP Client

Every MCP client lets you register servers via JSON. Point them at the `oumi-mcp` script. Examples:

### Claude Desktop

On macOS, edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "oumi": {
      "command": "oumi-mcp"
    }
  }
}
```

Then fully restart Claude Desktop. On Linux the path is `~/.config/Claude/claude_desktop_config.json`; on Windows it's `%APPDATA%\Claude\claude_desktop_config.json`.

### Claude Code

From the project directory:

```bash
claude mcp add oumi oumi-mcp
```

Or add it manually to `~/.claude.json` under `mcpServers`.

### Cursor

Settings → **MCP Servers** → add:

```json
{
  "oumi": {
    "command": "oumi-mcp"
  }
}
```

### Using a Specific Python Environment

If `oumi-mcp` isn't on your client's `PATH`, use an absolute path or wrap it in the environment where Oumi is installed:

```json
{
  "mcpServers": {
    "oumi": {
      "command": "/Users/you/.venvs/oumi/bin/oumi-mcp"
    }
  }
}
```

````{tip}
`oumi-mcp` needs credentials (`HF_TOKEN`, `WANDB_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`, etc.) in its process environment. The most reliable way is the MCP config's own `env` block, which every stdio-launching client supports:

```json
{
  "mcpServers": {
    "oumi": {
      "command": "oumi-mcp",
      "env": {
        "HF_TOKEN": "hf_...",
        "WANDB_API_KEY": "..."
      }
    }
  }
}
```

Shell rc files (`~/.zshrc`, `~/.bashrc`) only cover terminal-launched clients like Claude Code — macOS GUI apps (Claude Desktop, Cursor from the Dock) are started by `launchd` and never read them. Don't commit secrets inside the client config; reference them from a local secret store or keep the config out of version control.
````

## What's Exposed

The server surfaces two kinds of MCP primitives:

### Tools

Assistant-callable functions. The assistant should call `get_started` first — it returns a detailed catalog, path rules, and the recommended order of operations.

| Tool | Purpose |
|------|---------|
| `get_started` | **Call first.** Returns the full tool catalog, path rules, and workflow guidance. |
| `search_configs` | Fuzzy-search the ~500 bundled YAML configs by path, filename, and content. |
| `get_config` | Fetch one config's path, model, dataset, and raw YAML — for use as a **reference**. |
| `list_categories` | Browse available config categories, model families, and API providers. |
| `validate_config` | Validate a local YAML config against its schema before launch. |
| `pre_flight_check` | Check HF auth, hardware, local paths, and provider setup before launching. |
| `run_oumi_job` | Execute an Oumi command locally or on cloud. Dry-run by default. |
| `get_job_status` | Snapshot status for a tracked job. |
| `get_job_logs` | Tail logs for a job. |
| `list_jobs` | Inventory of running and completed jobs (filter with `status=`). |
| `cancel_job` | Cancel a running job. |
| `stop_cluster` | Stop a cluster (preserves infra, reduces compute cost). |
| `down_cluster` | Fully delete a cluster (halts all billing). Destructive. |
| `get_docs` | Search indexed Oumi Python API docs. |
| `list_modules` | List modules available for `get_docs` searches. |

### Resources

Read-only content the assistant can fetch by URI.

**Workflow guidance** (MLE playbooks):

| URI | Content |
|-----|---------|
| `guidance://mle-workflow` | Overall ML engineering workflow with Oumi. |
| `guidance://mle-train` | Guidance for `oumi train`. |
| `guidance://mle-synth` | Guidance for `oumi synth`. |
| `guidance://mle-analyze` | Guidance for `oumi analyze`. |
| `guidance://mle-eval` | Guidance for `oumi evaluate`. |
| `guidance://mle-infer` | Guidance for `oumi infer`. |
| `guidance://cloud-launch` | Anatomy of a cloud job config and setup patterns. |
| `guidance://post-training` | Post-training steps (download weights, eval, teardown). |

**Job state** (live):

| URI | Content |
|-----|---------|
| `jobs://running` | JSON list of currently running jobs. |
| `jobs://completed` | JSON list of recent completed / failed / cancelled jobs. |
| `jobs://{job_id}/logs` | Plain-text log output for a specific job. |

## Path Handling

There are two path surfaces, and they do not overlap:

- **Bundled config library.** `search_configs`, `get_config`, and `list_categories` browse the ~500 YAML files shipped with the `oumi` package. `get_config` accepts substring queries against those config paths (e.g. `"llama3_1/sft/8b_lora"`); it is read-only and its results should be treated as **references** — copy/adapt them into your own project, don't pass library paths to path-sensitive tools.
- **Your project files.** `run_oumi_job`, `validate_config`, and `pre_flight_check` take local filesystem paths (absolute, or relative to `client_cwd`). They do not resolve `oumi://` URIs or library config names — you must point them at a YAML file that exists on disk.

Assistants handle `client_cwd` automatically — `get_started` returns the full rules the model needs, and the assistant passes your project root on every path-sensitive call. You don't configure anything.

## Safety

Because `run_oumi_job` executes real commands, the server is deliberately cautious:

- **Dry-run by default.** `run_oumi_job` previews the command and exits unless the caller explicitly passes `dry_run=False`.
- **Pre-flight on cloud launches.** Cloud runs invoke `pre_flight_check` automatically (HF auth, gated repo access, hardware, local paths, SkyPilot compatibility). Blocking issues prevent launch.
- **Strict YAML parsing.** `run_oumi_job` rejects configs that fail schema validation.
- **Destructive actions require confirmation.** `down_cluster` requires a `confirm=True` flag and a typed confirmation string; `cancel_job` has a `force` flag that's opt-in.

An assistant that has been told to "just run it" still goes through these gates — use that to your advantage when delegating long-running work.

## A Typical Session

Here's what a full cloud-training flow looks like through a well-behaved assistant. You don't type any of this; the assistant does.

1. `get_started()` — fetch the current tool catalog and workflow.
2. `search_configs(query=["llama", "lora", "8b"])` — find candidate configs.
3. `get_config(path=...)` — inspect the most promising one as a reference.
4. Build a **job config** (resources + setup + run block) for the target cloud — see `guidance://cloud-launch`.
5. `validate_config(config="configs/my_job.yaml", task_type="job", client_cwd=...)` — catch schema errors early.
6. `pre_flight_check(config="configs/my_job.yaml", client_cwd=..., cloud="gcp")` — catch auth / hardware / path issues.
7. `run_oumi_job(config_path="configs/my_job.yaml", command="train", client_cwd=..., cloud="gcp", cluster_name="exp-1")` — dry-run first, then re-call with `dry_run=False` after you confirm.
8. `get_job_status(job_id=..., cloud="gcp", cluster_name="exp-1")` — poll.
9. `stop_cluster(cloud="gcp", cluster_name="exp-1")` — pause to save cost while you review results, or `down_cluster(...)` to shut everything down.

For local runs, drop the `cloud`/`cluster_name` arguments and point at a training config directly — there's no job-config requirement.

## Troubleshooting

### The assistant says it can't see the Oumi tools

- Run `oumi-mcp` in a terminal to confirm the process starts without errors. It has no CLI flags — it blocks on stdio and logs to stderr; kill it with `Ctrl-C`.
- Check the client's MCP logs. Claude Desktop logs to `~/Library/Logs/Claude/mcp*.log` on macOS.
- Verify the client is spawning `oumi-mcp` with the Python environment where Oumi is installed. If `which oumi-mcp` only works inside a virtualenv, pass an absolute path in the client config.
- Fully restart the client after editing its config.

### Logs go where?

The server uses stdio: MCP protocol traffic is on stdout, and Oumi's own logs are written to stderr at `INFO` level. Most clients surface stderr in their MCP log file. To see logs interactively, launch the server in a terminal (`oumi-mcp`) and watch stderr — the client won't start its own instance if one is already attached, so use this only for ad-hoc inspection, not as a permanent setup.

### Cloud launches fail pre-flight

Run `sky check` and follow the provider-specific setup in {doc}`/user_guides/launch/launch`. `pre_flight_check` surfaces the same issues but `sky check` gives the authoritative diagnostic.

### A job tool returns "job_id not found"

Call `list_jobs` to see every job the server currently knows about. For cloud jobs it queries the launcher live, so anything still alive on the provider shows up with its real job ID — pass that ID (plus `cloud` and `cluster_name`) to `get_job_status` / `get_job_logs`. Local-job logs are always on disk at `~/.oumi/mcp/job-logs/<job_id>/`.

## Under the Hood

- Source lives under {gh}`src/oumi/mcp/`.
- Tool implementations are registered in {gh}`src/oumi/mcp/server.py`.
- Guidance resources are simple strings under {gh}`src/oumi/mcp/prompts/`.
- Config discovery uses the same YAML library under {gh}`configs/` that the CLI does — no duplicate source of truth.

## See Also

- [Model Context Protocol spec](https://modelcontextprotocol.io/)
- {doc}`/cli/commands` — the CLI the server ultimately drives
- {doc}`/user_guides/launch/launch` — cloud job launch configuration
- {doc}`/user_guides/train/train` — training workflows the assistant can orchestrate
