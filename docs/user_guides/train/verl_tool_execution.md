# Nonblocking VERL tool execution

`OumiVerlTool` normally executes synchronous environments on the VERL agent-loop
thread. This preserves the existing behavior for short calls. A slow synchronous
executor can prevent unrelated rollouts on that event loop from making progress.

For synchronous database or I/O executors, set `execution_mode: thread` in the
**VERL tool config**, alongside `oumi_env_config`:

```yaml
config:
  oumi_env_config: path/to/environments.yaml
  execution_mode: thread
```

Use the same mode and environment config for every Oumi tool in a rollout.
The default is `sync`; no existing recipe changes behavior automatically.

Thread mode assigns a rollout to one of eight process-wide, single-thread
execution lanes. Its environments are built, driven, and closed on that lane.
This preserves SQLite connection affinity and serializes state mutations within
the rollout. Independent lanes progress concurrently without occupying the
agent-loop thread. Rollouts sharing a lane queue their work, so this is bounded
concurrency, not one thread per outstanding trajectory. Environment ownership and
per-row `create_kwargs` remain scoped to the rollout.

When the rollout's agent data is collected, teardown queues behind any in-flight
call. Cancelling an asyncio task cannot preempt a synchronous executor that has
already started; side effects may finish before teardown. Configure timeouts in
I/O executors so a hung request cannot hold a lane indefinitely.
On normal process exit, lanes drain queued calls and close any remaining owned
environments on their owning threads.

Only enable this mode when shared environments, callbacks, and custom executors
are safe to invoke from worker threads. In particular, do not use it to run
`SimulatedEnvironment` or an executor relying on an asyncio-bound inference
engine: such engines need native async execution on their owning event loop.
Python CPU-bound executors may remain limited by the GIL. Thread mode does not
change sampling, rewards, model computation, or batch-internal endpoint scheduling.

Measure the complete rollout workload before enabling it. A single cheap query
can be slower because of dispatch overhead, while many cold environments or
blocking calls can benefit substantially. Router-only benchmarks do not establish
an end-to-end GRPO training speedup.

## Reproduce the scheduling benchmark

From the repository root, with Oumi installed:

```bash
python scripts/benchmarks/benchmark_verl_tool_execution.py --output results.json
```

The benchmark uses real Oumi routers and SQLite environments, without model
inference. It compares identical queries with 1, 8, and 32 concurrent rollouts,
alternates execution-mode order over three repetitions, verifies matching outputs,
and saves every measurement. Each rollout performs one cold call and three warm
calls. Workloads include a recursive SQLite aggregation, a controlled 50 ms I/O
delay, and a cheap query to expose dispatch overhead. The parent router is built
before timing; cold measurements include per-rollout environment construction.

Use `warm_three_turn_seconds` to compare sustained tool execution separately from
initialization, and `max_event_loop_lag_seconds` to inspect event-loop stalls.
The delay workload models blocking I/O; it is not a measurement of a production
service. Validate complete rollout and training throughput on your own workload.
