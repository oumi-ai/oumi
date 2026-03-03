# Merge Cloud Path Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate duplicate path validation and false-positive errors by merging two cloud path walkers into one.

**Architecture:** Split `validate_paths` into `validate_paths_local` and `validate_paths_cloud`. The cloud version absorbs `_check_cloud_files` logic. Remove `cloud_file_checks` from the response model. Single error/warning loop in `_pre_flight_check`.

**Tech Stack:** Python, TypedDict, pytest, MCP server

---

### Task 1: Create `validate_paths_local` and `validate_paths_cloud`

**Files:**
- Modify: `src/oumi/mcp/preflight_service.py:131-400`

**Step 1: Rename and extract `validate_paths_local`**

Replace `validate_paths` (lines 131-181) with:

```python
def validate_paths_local(cfg: dict, base_dir: Path) -> dict[str, str]:
    """Validate config paths for local jobs.

    Walks ``_dir``/``_path``/``_file``/``_folder`` keys, resolves relative
    paths against *base_dir*, returns ``"ok"`` or ``"not_found"``.
    """
    paths: dict[str, str] = {}

    def _extract(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if isinstance(val, str) and key.endswith(
                    ("_dir", "_path", "_file", "_folder")
                ):
                    if _looks_like_hf_repo(val):
                        continue
                    _check(val)
                else:
                    _extract(val)
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)

    def _check(val: str) -> None:
        p = Path(val).expanduser()
        if not p.is_absolute():
            p = base_dir / p
            paths[f"{val} (resolved to {p})"] = "ok" if p.exists() else "not_found"
        else:
            paths[val] = "ok" if p.exists() else "not_found"

    _extract(cfg)
    return paths
```

**Step 2: Create `validate_paths_cloud`**

Add after `validate_paths_local`:

```python
def validate_paths_cloud(
    cfg: dict,
    config_path: Path,
    client_cwd: str,
    cloud: str,
) -> dict[str, str]:
    """Validate config paths for cloud jobs.

    For **job configs** (has ``resources``/``setup``/``run`` keys): validates
    ``file_mounts`` local sources and ``working_dir``.

    For **training configs**: walks ``_dir``/``_path``/``_file``/``_folder``
    keys and classifies each path.

    Statuses: ``"ok"``, ``"ok_remote"``, ``"not_found_warning"``,
    ``"local_machine_path_error"``, ``"missing_local_source"``,
    ``"unverifiable_remote"``, ``"working_dir_suspicious"``.
    """
    if not cloud or cloud == "local":
        return {}

    results: dict[str, str] = {}
    base_dir = Path(client_cwd)
    job_keys = {"resources", "setup", "run"}
    is_job_cfg = bool(job_keys.intersection(cfg.keys()))

    if is_job_cfg:
        # -- file_mounts --
        for _remote, local_src in (cfg.get("file_mounts") or {}).items():
            if not isinstance(local_src, str):
                continue
            expanded = Path(local_src).expanduser()
            if not expanded.is_absolute():
                expanded = base_dir / expanded
            results[local_src] = "ok" if expanded.exists() else "missing_local_source"

        # -- working_dir --
        wd = cfg.get("working_dir")
        if wd is not None and str(wd) != ".":
            wd_path = Path(str(wd)).expanduser()
            if not wd_path.is_absolute():
                wd_path = base_dir / wd_path
            if not wd_path.exists():
                results[f"working_dir: {wd}"] = "working_dir_suspicious"
    else:
        # -- training config: walk structured keys --
        def _extract(obj: object) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    if isinstance(val, str) and key.endswith(
                        ("_dir", "_path", "_file", "_folder")
                    ):
                        _classify(val)
                    else:
                        _extract(val)
            elif isinstance(obj, list):
                for item in obj:
                    _extract(item)

        def _classify(val: str) -> None:
            if not val or val.isspace():
                return
            if _looks_like_hf_repo(val) and not _FILE_EXT_RE.search(val):
                return
            p = Path(val)
            if p.is_absolute():
                if _is_local_machine_path(val):
                    results[val] = "local_machine_path_error"
                else:
                    results[val] = "unverifiable_remote"
            else:
                resolved = base_dir / val
                results[val] = "ok" if resolved.exists() else "not_found_warning"

        _extract(cfg)

    return results
```

**Step 3: Delete old functions**

Delete `_check_cloud_files` (lines ~307-333), `_check_job_config_files` (lines ~336-364), `_check_training_config_files` (lines ~367-398).

**Step 4: Run tests to see what breaks**

Run: `python -m pytest tests/unit/mcp/test_preflight_service.py tests/unit/mcp/test_cloud_file_checks.py -v`

---

### Task 2: Update `_pre_flight_check` consumer

**Files:**
- Modify: `src/oumi/mcp/preflight_service.py:868-944`

**Step 1: Replace the two-pass path validation**

Replace lines 868-913 (the `validate_paths` call + `cloud_file_checks` block) with:

```python
    if target_cloud:
        path_results = validate_paths_cloud(
            cfg, config_path, client_cwd, target_cloud
        )
    else:
        path_results = validate_paths_local(cfg, Path(client_cwd))

    for path_key, path_status in path_results.items():
        if path_status == "local_machine_path_error":
            errors.append(
                f"Local machine path '{path_key}' will not exist on the remote VM. "
                "Use a repo-relative path (e.g., 'data/...') that resolves from "
                "your working_dir."
            )
        elif path_status == "missing_local_source":
            errors.append(
                f"file_mounts source '{path_key}' does not exist locally. "
                "The file won't be copied to the remote VM."
            )
        elif path_status == "not_found" or path_status == "not_found_warning":
            warnings.append(
                f"Path '{path_key}' not found locally. "
                "Verify it will be available on the VM via file_mounts, "
                "working_dir, or setup_script."
            )
        elif path_status == "working_dir_suspicious":
            warnings.append(
                f"'{path_key}' does not exist locally. Use 'working_dir: .' "
                "(resolved to client_cwd at launch) or verify the path."
            )
        elif path_status == "unverifiable_remote":
            warnings.append(
                f"Remote path '{path_key}' can't be validated locally. "
                "Ensure it exists on the VM via setup_script or storage_mounts."
            )

    if target_cloud:
        skyignore_warnings = _check_skyignore(config_path.parent, path_results)
        warnings.extend(skyignore_warnings)
```

**Step 2: Remove `cloud_file_checks` from the result dict**

In the result construction (~line 928-944), remove:
```python
    if cloud_file_checks:
        result["cloud_file_checks"] = cloud_file_checks
```

**Step 3: Run tests**

Run: `python -m pytest tests/unit/mcp/test_preflight_service.py -v`

---

### Task 3: Remove `cloud_file_checks` from model

**Files:**
- Modify: `src/oumi/mcp/models.py:152-170`

**Step 1: Remove the field and docstring line**

Remove from `PreFlightCheckResponse`:
```python
    cloud_file_checks: NotRequired[dict[str, str]]
```

And its docstring entry:
```
        cloud_file_checks: Per-path cloud file delivery validation. ...
```

Update the `paths` docstring to include the new cloud statuses:
```
        paths: Config paths mapped to validation status: "ok", "not_found",
            "ok_remote", "not_found_warning", "local_machine_path_error",
            "missing_local_source", "unverifiable_remote", or
            "working_dir_suspicious".
```

**Step 2: Run type check**

Run: `pyright src/oumi/mcp/models.py src/oumi/mcp/preflight_service.py`

---

### Task 4: Update tests

**Files:**
- Modify: `tests/unit/mcp/test_preflight_service.py:53-106`
- Modify: `tests/unit/mcp/test_cloud_file_checks.py`

**Step 1: Update `test_preflight_service.py`**

Replace `TestValidatePaths` to use `validate_paths_local` for local tests and `validate_paths_cloud` for cloud tests. Update imports: replace `validate_paths` with `validate_paths_local, validate_paths_cloud`.

Local tests stay the same but call `validate_paths_local(cfg, tmp_path)`.

Cloud tests call `validate_paths_cloud(cfg, tmp_path / "config.yaml", str(tmp_path), "gcp")`.

Key behavioral change to test: `test_cloud_relative_not_found` should still return `not_found_warning`.

**Step 2: Update `test_cloud_file_checks.py`**

Replace `_check_cloud_files` import with `validate_paths_cloud`. Update all calls:
- `_check_cloud_files(cfg, path, "aws")` → `validate_paths_cloud(cfg, path, str(tmp_path), "aws")`
- Add `client_cwd` parameter to each call

In `TestPreFlightIntegration`: remove `cloud_file_checks` assertions, check `paths` dict instead.

In `TestTrainingConfigWrapping::test_relative_path_blocking`: change assertion from `"not_reachable_on_vm"` to `"not_found_warning"` (this is the key behavioral fix for issue #2).

**Step 3: Run full test suite**

Run: `python -m pytest tests/unit/mcp/ -v`

**Step 4: Commit**

```
refactor(mcp): merge cloud path validation into two clean functions

Replace validate_paths + _check_cloud_files with validate_paths_local and
validate_paths_cloud. Eliminates duplicate error messages, downgrades
training config relative paths from errors to warnings, and resolves
file_mounts relative paths against client_cwd.
```

---

## Verification

1. `python -m pytest tests/unit/mcp/test_preflight_service.py -v` — path validation tests pass
2. `python -m pytest tests/unit/mcp/test_cloud_file_checks.py -v` — cloud file check tests pass
3. `python -m pytest tests/unit/mcp/ -v` — full MCP suite passes
4. `pyright src/oumi/mcp/models.py src/oumi/mcp/preflight_service.py` — no type errors
