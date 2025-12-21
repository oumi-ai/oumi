# Simplified Distributed Training CLI

## Summary

Added `--distributed` flag to `oumi train` for automatic multi-GPU training, and deprecated the separate `oumi distributed` command.

## New Files Created

- **`src/oumi/core/cluster.py`** - Centralized cluster detection logic with `ClusterInfo`, `WorldInfo`, `ClusterBackend` enum, and `detect_cluster_info()` function

- **`tests/unit/core/test_cluster.py`** - Tests for `ClusterInfo` and `WorldInfo` classes

## Modified Files

### `src/oumi/utils/distributed_utils.py`
Added:
- `is_using_torchrun()` - detects torchrun via `TORCHELASTIC_RUN_ID`
- `is_under_distributed_launcher()` - detects torchrun or accelerate

### `src/oumi/cli/train.py`
Added:
- `--distributed` / `-d` flag for auto-launching torchrun
- `_handle_distributed_training()` function with behavior matrix:
  - Already under launcher → no-op
  - `--distributed` + multi-GPU → re-exec with torchrun
  - `--distributed` + single-GPU → log info, continue
  - No `--distributed` + multi-GPU → warn about unused GPUs

### `src/oumi/cli/distributed_run.py`
Updated:
- Added deprecation warnings for `torchrun` and `accelerate` commands
- Made imports lazy to avoid CLI startup regression
- Imports `ClusterInfo` and `detect_cluster_info` from `cluster.py`

### `tests/unit/cli/test_cli_distributed_run.py`
Updated mock fixtures to work with new module structure

### `tests/unit/utils/test_distributed_utils.py`
Added tests for new launcher detection functions

## New Usage

```bash
# Single GPU - just works
oumi train -c config.yaml

# Multi-GPU - auto-launches torchrun
oumi train --distributed -c config.yaml
oumi train -d -c config.yaml  # short form

# Power users can still use torchrun directly
torchrun --nproc-per-node=4 -m oumi train -c config.yaml
```

## Behavior Matrix

| Command | Under Launcher | GPUs | Behavior |
|---------|---------------|------|----------|
| `oumi train -c config.yaml` | No | 1 | Normal training |
| `oumi train -c config.yaml` | No | 4 | WARNING: "Multiple GPUs detected but --distributed not set" |
| `oumi train --distributed -c config.yaml` | No | 1 | Log "Single GPU detected", normal training |
| `oumi train --distributed -c config.yaml` | No | 4 | Re-exec with torchrun |
| `oumi train -c config.yaml` | Yes | any | Normal distributed training |
| `oumi train --distributed -c config.yaml` | Yes | any | No-op, normal distributed training |

## Deprecated Commands

The following commands now emit deprecation warnings:
- `oumi distributed torchrun` → Use `oumi train --distributed` or `torchrun -m oumi train` directly
- `oumi distributed accelerate` → Use `oumi train --distributed` or `accelerate launch -m oumi train` directly
