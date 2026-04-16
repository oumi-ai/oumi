# MCP Server

```{admonition} Experimental
:class: warning
The Oumi MCP server is under active development (Phase 1). Tools, resources, and prompts may change as we iterate on the integration with MCP-capable assistants.
```

Oumi ships an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that lets AI assistants — e.g. Claude Desktop, Claude Code, Cursor — discover Oumi configs, run training/eval/inference jobs, and read workflow guidance without leaving the chat interface. It's installed as a separate extra and launched as a standalone process.

## Installation

```bash
pip install "oumi[mcp]"
```

This pulls in `fastmcp`, the `mcp` package, and `httpx`. Once installed, a new console script is available:

```bash
oumi-mcp          # starts the MCP server on stdio
python -m oumi.mcp   # equivalent
```

## Connecting from an MCP Client

Most MCP-capable clients expect a JSON entry describing how to launch the server. Point them at the `oumi-mcp` script:

```json
{
  "mcpServers": {
    "oumi": {
      "command": "oumi-mcp"
    }
  }
}
```

Exact placement depends on the client — Claude Desktop reads `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS; other clients use similar conventions. Refer to your client's MCP docs for the exact path.

## What's Exposed

The server surfaces three kinds of MCP primitives:

**Tools** (assistant-callable functions)

| Tool                 | Purpose                                                      |
|----------------------|--------------------------------------------------------------|
| `search_configs`     | Fuzzy-search the ~500 Oumi YAML configs by path + content    |
| `get_config`         | Fetch full details (path, model, dataset, raw YAML) for one config |
| `launch_job`         | Launch an Oumi job locally or on a cloud provider            |
| `poll_status`        | Poll a running job's status                                  |
| `stop_cluster` / `down_cluster` | Manage SkyPilot clusters                            |
| `cancel_job`         | Cancel a running job                                         |
| `fetch_logs`         | Retrieve logs for a job                                      |
| `list_running_jobs` / `list_completed_jobs` | Inventory of tracked jobs                 |

**Resources** (workflow guidance strings the assistant can read)

- `guidance://mle-workflow` — overall ML engineering workflow
- `guidance://mle-train`, `mle-synth`, `mle-analyze`, `mle-eval`, `mle-infer` — per-command guidance
- `guidance://cloud-launch` — cloud job anatomy and setup patterns
- `guidance://post-training` — post-training steps (download weights, eval, teardown)

**Prompts** (pre-built prompt templates for common tasks). See the source under `oumi.mcp.prompts` for the current list.

## Path Rules (Important)

Because the server executes real commands against the user's machine or cloud account, it's strict about paths:

- Every path-sensitive tool requires `client_cwd` — the user's project root.
- Config paths may be absolute or relative to `client_cwd`.
- **Local jobs**: the subprocess runs from `client_cwd`; paths inside the YAML resolve against that directory.
- **Cloud jobs**: `client_cwd` becomes `working_dir` on the remote VM. Use repo-relative paths in the YAML — never local-machine absolute paths.

Assistants should always call the `get_started()` tool first (returned by the server) to retrieve the up-to-date tool catalog and workflow before doing anything else.

## Debugging

`oumi-mcp` speaks stdio MCP by default and logs to stderr. To inspect traffic, run it from a terminal and set `OUMI_LOG_LEVEL=DEBUG`:

```bash
OUMI_LOG_LEVEL=DEBUG oumi-mcp
```

Then configure the client to spawn it with the same environment.

## See Also

- [Model Context Protocol spec](https://modelcontextprotocol.io/)
- {doc}`/cli/commands` — what the underlying `oumi` CLI can do
- {doc}`/user_guides/launch/launch` — details on cloud job launching
