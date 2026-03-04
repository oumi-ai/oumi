# Merge Cloud Path Validation Into Two Clean Functions

## Problem

Two separate config walkers (`validate_paths` cloud branch and `_check_cloud_files`) extract the same `_dir/_path/_file/_folder` keys from training configs with different logic, producing conflicting statuses and duplicate error messages for the same path. Additionally, `_check_training_config_files` flags ALL relative paths as hard errors (`not_reachable_on_vm`), even though a wrapping job config's `working_dir` will typically deliver them to the VM.

## Design

### `validate_paths_local(cfg, base_dir) -> dict[str, str]`

Extracted from the local branch of the current `validate_paths`. Walks `_dir/_path/_file/_folder` keys, resolves relative paths against `base_dir`, returns `valid` or `not_found`. No behavior change.

### `validate_paths_cloud(cfg, config_path, client_cwd, cloud) -> dict[str, str]`

Merges the cloud branch of `validate_paths` + all of `_check_cloud_files` into a single pass.

**Job configs** (has `resources`/`setup`/`run` keys):
- Checks `file_mounts` sources exist locally, resolved against `client_cwd` for relative paths
- Validates `working_dir` (resolved against `client_cwd` if relative, skip `.`)
- Statuses: `valid`, `missing_local_source`, `working_dir_suspicious`

**Training configs**:
- Walks `_dir/_path/_file/_folder` keys (same extractor as local)
- Skips HF repos via `_looks_like_hf_repo`
- Local-machine absolute paths -> `local_machine_path_error`
- Remote absolute paths -> `unverifiable_remote`
- Relative paths -> `not_found_warning` (warning, not error)
- Statuses: `local_machine_path_error`, `unverifiable_remote`, `not_found_warning`, `valid`

### `_pre_flight_check` changes

- Local: `path_results = validate_paths_local(cfg, Path(client_cwd))`
- Cloud: `path_results = validate_paths_cloud(cfg, config_path, client_cwd, target_cloud)`
- Single error/warning generation loop over `path_results`
- Remove the separate `cloud_file_checks` second pass and response key; fold into `paths`

### Deletions

- `_check_cloud_files`
- `_check_job_config_files`
- `_check_training_config_files`
- Cloud branch of `validate_paths`

### Key behavioral changes

1. Training config relative paths become `not_found_warning` (warning) instead of `not_reachable_on_vm` (error). Eliminates false positives.
2. One status per path, one error/warning loop. No more duplicate messages.
3. `file_mounts` relative sources resolve against `client_cwd` instead of process cwd.

### Response shape change

`cloud_file_checks` key is removed from `PreFlightCheckResponse`. Its statuses are folded into the `paths` dict. The `cloud_file_checks` field on the TypedDict model is removed.
