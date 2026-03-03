"""Pre-flight check service for Oumi MCP Server.

Validates HF auth, gated repo access, hardware compatibility, cloud
credentials, dataset accessibility, and cloud file delivery before launch.
"""

import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, cast

from huggingface_hub import auth_check, whoami
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    HFValidationError,
    RepositoryNotFoundError,
)
from packaging.version import Version

from oumi.mcp.config_service import (
    get_all_configs,
    get_package_version,
    load_yaml_strict,
    resolve_config_path,
)
from oumi.mcp.config_service import (
    search_configs as search_configs_service,
)
from oumi.mcp.constants import (
    HARDWARE_PACKAGES,
    HF_API_TIMEOUT_SECONDS,
    MIN_CC_BF16,
    MIN_CC_FLASH_ATTN,
    MIN_TORCH_VERSION_COMPILE,
    MIN_TORCH_VERSION_SDPA,
)
from oumi.mcp.models import (
    CloudReadiness,
    HardwareInfo,
    PreFlightCheckResponse,
)

try:
    import torch as _torch
except (ImportError, ModuleNotFoundError):
    _torch = None
torch = _torch

try:
    import sky as _sky
    from sky import check as _sky_check
    from sky.clouds.cloud import CloudCapability as _CloudCapability
except (ImportError, ModuleNotFoundError):
    _sky = None
    _sky_check = None
    _CloudCapability = None
sky = _sky
sky_check = _sky_check
CloudCapability = _CloudCapability

logger = logging.getLogger(__name__)

_FILE_EXT_RE = re.compile(
    r"\.(jsonl|json|csv|parquet|txt|arrow|bin|safetensors|pt|gguf|yaml|yml)$",
    re.IGNORECASE,
)


def get_gpu_info() -> dict[str, Any]:
    """Return a dict describing available GPU/accelerator hardware."""
    info: dict[str, Any] = {
        "accelerator_type": "none",
        "accelerator_count": 0,
        "accelerators": [],
        "gpu_name": None,
        "gpu_memory_bytes": None,
    }
    try:
        if torch is not None and torch.cuda.is_available():
            info["accelerator_type"] = "cuda"
            count = torch.cuda.device_count()
            info["accelerator_count"] = count
            if count > 0:
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_memory_bytes"] = props.total_mem
                info["accelerators"] = [
                    {
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                ]
        elif (
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            info["accelerator_type"] = "mps"
            info["accelerator_count"] = 1
    except Exception:
        pass
    return info


def _looks_like_hf_repo(val: str) -> bool:
    """Return True if *val* looks like an HF repo ID (org/name)."""
    return bool(val) and val.count("/") == 1 and not val.startswith(("/", ".", "~"))


def _is_local_machine_path(path_str: str) -> bool:
    """Return True if *path_str* is a local machine absolute path.

    Detects paths rooted at /Users/, /home/<local-user>/, or matching
    Path.home(). Remote absolute paths (e.g. /home/ubuntu/...) are NOT
    considered local.
    """
    p = Path(path_str)
    if not p.is_absolute():
        return False
    home = Path.home()
    if p == home or str(p).startswith(str(home) + "/"):
        return True
    if path_str.startswith("/Users/"):
        return True
    return False


def validate_paths_local(cfg: dict, base_dir: Path) -> dict[str, str]:
    """Validate config paths for local jobs.

    Walks ``_dir``/``_path``/``_file``/``_folder`` keys, resolves relative
    paths against *base_dir*, returns ``"valid"`` or ``"not_found"``.
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
            paths[f"{val} (resolved to {p})"] = "valid" if p.exists() else "not_found"
        else:
            paths[val] = "valid" if p.exists() else "not_found"

    _extract(cfg)
    return paths


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

    Statuses: ``"valid"``, ``"valid_remote"``, ``"not_found_warning"``,
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
        for _remote, local_src in (cfg.get("file_mounts") or {}).items():
            if not isinstance(local_src, str):
                continue
            expanded = Path(local_src).expanduser()
            if not expanded.is_absolute():
                expanded = base_dir / expanded
            results[local_src] = (
                "valid" if expanded.exists() else "missing_local_source"
            )

        wd = cfg.get("working_dir")
        if wd is not None and str(wd) != ".":
            wd_path = Path(str(wd)).expanduser()
            if not wd_path.is_absolute():
                wd_path = base_dir / wd_path
            if not wd_path.exists():
                results[f"working_dir: {wd}"] = "working_dir_suspicious"
    else:

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
                results[val] = "valid" if resolved.exists() else "not_found_warning"

        _extract(cfg)

    return results


def validate_datasets(cfg: dict, client_cwd: str = "") -> dict[str, str]:
    """Validate dataset accessibility for each dataset in the config.

    Mirrors Oumi's dataset resolution chain:
    1. REGISTRY.get_dataset(name) → found in registry
    2. dataset_path set → check local existence (resolved against client_cwd)
    3. datasets.load_dataset_builder(name) → HF Hub metadata probe (no download)

    Returns a dict mapping dataset identifiers to status strings:
    ``"ok_registry"``, ``"ok_local"``, ``"ok_hub"``, ``"not_found"``,
    ``"warning_timeout"``.
    """
    data = cfg.get("data") or {}
    results: dict[str, str] = {}
    base_dir = Path(client_cwd) if client_cwd else Path.cwd()

    for split in ("train", "eval", "validation", "test"):
        split_cfg = data.get(split) or {}
        for ds in split_cfg.get("datasets") or []:
            ds_name = ds.get("dataset_name", "")
            ds_path = ds.get("dataset_path", "")

            if not ds_name and not ds_path:
                continue

            key = ds_name or ds_path

            if key in results:
                continue

            if ds_name:
                try:
                    from oumi.core.registry import REGISTRY

                    reg_result = REGISTRY.get_dataset(ds_name)
                    if reg_result is not None:
                        results[key] = "ok_registry"
                        continue
                except Exception:
                    pass

            if ds_path:
                p = Path(ds_path).expanduser()
                if not p.is_absolute():
                    p = (base_dir / p).resolve()
                if p.exists():
                    results[key] = "ok_local"
                    continue

            if ds_name:
                try:
                    import datasets

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(datasets.load_dataset_builder, ds_name)
                        future.result(timeout=5)
                    results[key] = "ok_hub"
                    continue
                except TimeoutError:
                    results[key] = "warning_timeout"
                    continue
                except Exception:
                    pass

            results[key] = "not_found"

    return results


def _check_skyignore(config_dir: Path, path_results: dict[str, str]) -> list[str]:
    """Check if a .skyignore file might exclude files needed by the config.

    Walks up from *config_dir* looking for ``.skyignore``. If found, parses
    its patterns and warns if any config paths appear to match.
    """
    warnings: list[str] = []

    skyignore_path: Path | None = None
    search = config_dir.resolve()
    for _ in range(10):  # limit depth
        candidate = search / ".skyignore"
        if candidate.is_file():
            skyignore_path = candidate
            break
        parent = search.parent
        if parent == search:
            break
        search = parent

    if skyignore_path is None:
        return warnings

    try:
        patterns = [
            line.strip()
            for line in skyignore_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return warnings

    if not patterns:
        return warnings

    warnings.append(
        f"Found .skyignore at {skyignore_path} — verify it doesn't exclude "
        "files needed on the remote VM."
    )

    for path_key in path_results:
        raw_path = path_key.split(" (resolved to")[0]
        for pattern in patterns:
            bare = pattern.rstrip("/")
            if raw_path == bare or raw_path.startswith(bare + "/"):
                warnings.append(
                    f"Config path '{raw_path}' may be excluded by "
                    f".skyignore pattern '{pattern}'"
                )
                break

    return warnings


def get_repos(cfg: dict) -> dict[str, set[str]]:
    """Extract all HF repo IDs with their repo types from a parsed config."""
    repos: dict[str, set[str]] = {}

    def add(repo_id: str, repo_type: str) -> None:
        if _looks_like_hf_repo(repo_id):
            repos.setdefault(repo_id, set()).add(repo_type)

    model = cfg.get("model") or {}
    add(model.get("model_name", ""), "model")
    add(model.get("tokenizer_name", ""), "model")

    data = cfg.get("data") or {}
    for split in ("train", "eval", "validation", "test"):
        split_cfg = data.get(split) or {}
        for ds in split_cfg.get("datasets") or []:
            add(ds.get("dataset_name", ""), "dataset")
            ds_kwargs = ds.get("dataset_kwargs") or {}
            add(ds_kwargs.get("hf_dataset_path", "") or "", "dataset")

    training = cfg.get("training") or {}
    add(training.get("teacher_model_name_or_path", ""), "model")

    gold = training.get("gold") or {}
    add(gold.get("teacher_model_name_or_path", ""), "model")

    return repos


def _empty_hardware() -> HardwareInfo:
    """Return a default HardwareInfo with no accelerator detected."""
    return {
        "accelerator_type": "none",
        "accelerator_count": 0,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "compute_capability": None,
        "cuda_version": None,
        "packages": {},
    }


def _empty_cloud_readiness() -> CloudReadiness:
    """Return a default CloudReadiness with nothing checked."""
    return {
        "sky_installed": False,
        "enabled_clouds": [],
        "target_cloud_ready": None,
        "target_cloud": "",
    }


def _skypilot_version_label() -> str:
    """Return a short, user-facing SkyPilot version label."""
    version = get_package_version("skypilot") or get_package_version("sky")
    return f"skypilot={version}" if version else "skypilot=unknown"


def _compat_warning(message: str) -> str:
    """Format a version-aware SkyPilot compatibility warning."""
    return f"SkyPilot API compatibility issue ({_skypilot_version_label()}): {message}"


def _compat_error(message: str) -> str:
    """Format a version-aware SkyPilot compatibility error."""
    return f"SkyPilot API compatibility error ({_skypilot_version_label()}): {message}"


def _cloud_names(values: list[Any]) -> list[str]:
    """Normalize SkyPilot cloud objects/strings to canonical cloud names."""
    names: list[str] = []
    for value in values:
        name = str(value).strip()
        if name:
            names.append(name.upper())
    return sorted(set(names))


def _get_compute_capability(sky: Any) -> Any:
    """Return SkyPilot compute capability enum value."""
    try:
        if CloudCapability is None:
            raise RuntimeError("CloudCapability is unavailable.")
        return CloudCapability.COMPUTE
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve CloudCapability.COMPUTE from SkyPilot."
        ) from exc


def _get_enabled_clouds(sky: Any, sky_check: Any) -> list[Any]:
    """Return enabled clouds using SkyPilot's old/new check API variants."""
    get_enabled = getattr(sky_check, "get_cached_enabled_clouds_or_refresh", None)
    if not callable(get_enabled):
        raise RuntimeError(
            "sky.check.get_cached_enabled_clouds_or_refresh() is unavailable."
        )
    try:
        return list(cast(Iterable[Any], get_enabled()))
    except TypeError as exc:
        if "capability" not in str(exc):
            raise
        capability = _get_compute_capability(sky)
        return list(cast(Iterable[Any], get_enabled(capability)))


def _target_cloud_ready(
    sky: Any,
    sky_check: Any,
    *,
    target_cloud: str,
    enabled_clouds: list[str],
) -> bool:
    """Check if a target cloud is ready across SkyPilot API versions."""
    target_name = target_cloud.upper()
    if target_name in enabled_clouds:
        return True

    check_capability = getattr(sky_check, "check_capability", None)
    if callable(check_capability):
        capability = _get_compute_capability(sky)
        try:
            status = check_capability(capability, quiet=True, clouds=[target_cloud])
        except TypeError:
            status = check_capability(capability, clouds=[target_cloud])
        if isinstance(status, dict):
            ready_clouds: list[str] = []
            for cloud_list in status.values():
                if isinstance(cloud_list, list):
                    ready_clouds.extend(str(cloud).upper() for cloud in cloud_list)
            return target_name in ready_clouds
        return False

    check_one_cloud = getattr(sky_check, "check_one_cloud", None)
    if callable(check_one_cloud):
        cloud_obj = sky.CLOUD_REGISTRY.from_str(target_cloud)  # type: ignore[attr-defined]
        return bool(check_one_cloud(cloud_obj))

    raise RuntimeError(
        "No supported targeted cloud check API found "
        "(expected sky.check.check_capability or sky.check.check_one_cloud)."
    )


def check_cloud_readiness(
    target_cloud: str = "",
) -> tuple[list[str], list[str], CloudReadiness]:
    """Check SkyPilot cloud credentials and readiness.

    Uses SkyPilot's cloud-check APIs to discover which clouds have valid
    credentials (uses cache, refreshes if needed).

    If *target_cloud* is provided (e.g. ``"gcp"``), additionally validates
    that specific cloud via supported SkyPilot targeted check APIs and reports
    a blocking error if credentials are invalid.

    Returns ``(errors, warnings, cloud_readiness)``.
    """
    errors: list[str] = []
    warnings: list[str] = []
    result = _empty_cloud_readiness()

    if sky is None or sky_check is None:
        result["sky_installed"] = False
        if target_cloud:
            errors.append(
                "SkyPilot (sky) is not installed. "
                "Install it with: pip install 'skypilot-nightly[all]'"
            )
        return errors, warnings, result

    result["sky_installed"] = True

    try:
        enabled = _get_enabled_clouds(sky, sky_check)
    except RuntimeError as exc:
        # SkyPilot may raise RuntimeError when no clouds are enabled. Treat
        # those as non-blocking for broad checks; treat other runtime failures
        # as compatibility errors.
        msg = str(exc).lower()
        if "no cloud access" in msg or "no enabled cloud" in msg:
            enabled = []
        else:
            if target_cloud:
                errors.append(_compat_error(f"Target cloud check failed: {exc}"))
                result["target_cloud"] = target_cloud
                result["target_cloud_ready"] = False
            else:
                warnings.append(
                    _compat_warning(f"Failed to check cloud credentials: {exc}")
                )
            return errors, warnings, result
    except Exception as exc:
        message = _compat_warning(f"Failed to check cloud credentials: {exc}")
        if target_cloud:
            errors.append(_compat_error(f"Target cloud check failed: {exc}"))
            result["target_cloud"] = target_cloud
            result["target_cloud_ready"] = False
        else:
            warnings.append(message)
        return errors, warnings, result

    enabled_names = _cloud_names(enabled)
    result["enabled_clouds"] = enabled_names

    if target_cloud:
        result["target_cloud"] = target_cloud
        try:
            ok = _target_cloud_ready(
                sky,
                sky_check,
                target_cloud=target_cloud,
                enabled_clouds=enabled_names,
            )
        except Exception as exc:
            result["target_cloud_ready"] = False
            errors.append(_compat_error(f"Target cloud check failed: {exc}"))
            return errors, warnings, result

        if ok:
            result["target_cloud_ready"] = True
        else:
            result["target_cloud_ready"] = False
            errors.append(
                f"Cloud '{target_cloud}' is not ready. "
                f"Enabled clouds: {enabled_names}. "
                "Run 'sky check' for setup instructions."
            )

    if not enabled_names:
        warnings.append(
            "No cloud providers have valid credentials. "
            "Run 'sky check' to configure cloud access."
        )

    return errors, warnings, result


def check_hardware(cfg: dict) -> tuple[list[str], list[str], HardwareInfo]:
    """Detect local hardware and check compatibility with config requirements."""

    errors: list[str] = []
    warnings: list[str] = []

    gpu_info = get_gpu_info()
    accel_type = gpu_info.get("accelerator_type", "none")
    has_gpu = accel_type in ("cuda", "mps")
    cc_str: str | None = None
    cc: float = 0.0

    if accel_type == "cuda" and gpu_info.get("accelerators"):
        cc_str = gpu_info["accelerators"][0].get("compute_capability")
        if cc_str:
            try:
                cc = float(cc_str)
            except ValueError:
                pass

    packages: dict[str, str] = {}
    for pkg in HARDWARE_PACKAGES:
        ver = get_package_version(pkg)
        if ver:
            packages[pkg] = ver
    torch_ver = packages.get("torch")

    hardware = _empty_hardware()
    hardware["accelerator_type"] = accel_type
    hardware["accelerator_count"] = gpu_info.get("accelerator_count", 0)
    hardware["gpu_name"] = gpu_info.get("gpu_name")
    hardware["gpu_memory_gb"] = (
        round(gpu_info["gpu_memory_bytes"] / (1024**3), 1)
        if gpu_info.get("gpu_memory_bytes")
        else None
    )
    hardware["compute_capability"] = cc_str
    hardware["cuda_version"] = gpu_info.get("cuda_version")
    hardware["packages"] = packages

    model_cfg = cfg.get("model") or {}
    attn_impl = model_cfg.get("attn_implementation", "")
    peft_cfg = cfg.get("peft") or {}
    training_cfg = cfg.get("training") or {}
    ds_cfg = cfg.get("deepspeed") or {}
    fsdp = training_cfg.get("fsdp") or ""
    dtype = model_cfg.get("torch_dtype_str", "") or training_cfg.get("dtype", "")
    uses_bf16 = "bf16" in str(dtype).lower()

    if attn_impl == "flash_attention_2" and "flash-attn" not in packages:
        errors.append(
            "Config requires flash_attention_2 but 'flash-attn' is not installed"
        )

    if peft_cfg.get("q_lora") and "bitsandbytes" not in packages:
        errors.append("Config requires QLoRA but 'bitsandbytes' is not installed")

    if ds_cfg.get("enable_deepspeed") and "deepspeed" not in packages:
        errors.append("Config enables DeepSpeed but 'deepspeed' is not installed")

    if torch_ver:
        tv = Version(torch_ver)
        if attn_impl == "sdpa" and tv < Version(MIN_TORCH_VERSION_SDPA):
            errors.append(
                f"Config requires SDPA but torch {torch_ver} < {MIN_TORCH_VERSION_SDPA}"
            )
        if training_cfg.get("compile") and tv < Version(MIN_TORCH_VERSION_COMPILE):
            errors.append(
                f"Config requires torch.compile but torch {torch_ver} "
                f"< {MIN_TORCH_VERSION_COMPILE}"
            )

    if (fsdp or ds_cfg.get("enable_deepspeed")) and accel_type == "none":
        warnings.append(
            "Config uses FSDP/DeepSpeed but no GPU detected locally. "
            "This is fine if targeting a remote cluster."
        )

    if fsdp and accel_type == "mps":
        warnings.append("FSDP is not supported on MPS (Apple Silicon)")

    if attn_impl == "flash_attention_2" and accel_type == "mps":
        warnings.append("flash_attention_2 is not supported on MPS (Apple Silicon)")

    if training_cfg.get("fused_optimizer") and not has_gpu:
        warnings.append("Config uses fused optimizer but no GPU detected locally")

    if uses_bf16 and not has_gpu:
        warnings.append(
            "Config uses bf16 but no GPU detected locally. "
            "This is fine if targeting a remote cluster."
        )

    if accel_type == "cuda" and cc > 0:
        if uses_bf16 and cc < MIN_CC_BF16:
            warnings.append(
                f"Config uses bf16 but GPU compute capability {cc_str} "
                f"< {MIN_CC_BF16} (Ampere). bf16 may not be natively supported."
            )
        if attn_impl == "flash_attention_2" and cc < MIN_CC_FLASH_ATTN:
            warnings.append(
                f"Config uses flash_attention_2 but GPU compute capability "
                f"{cc_str} < {MIN_CC_FLASH_ATTN} (Ampere). "
                f"Flash attention 2 requires Ampere or newer."
            )

    return errors, warnings, hardware


def _pre_flight_check(
    config: str, client_cwd: str, cloud: str = ""
) -> PreFlightCheckResponse:
    """Run pre-flight checks (internal implementation)."""
    errors: list[str] = []
    warnings: list[str] = []
    repo_access: dict[str, str] = {}

    config_path, path_error = resolve_config_path(config, client_cwd)
    if path_error:
        errors.append(path_error)
        return {
            "blocking": True,
            "summary": f"BLOCKED: {path_error}",
            "hf_authenticated": False,
            "repo_access": {},
            "hardware": _empty_hardware(),
            "cloud_readiness": _empty_cloud_readiness(),
            "errors": errors,
            "warnings": [],
            "paths": {},
            "skypilot_compat_issue": False,
        }

    cfg, load_error = load_yaml_strict(config_path)
    if load_error:
        errors.append(load_error)
        return {
            "blocking": True,
            "summary": f"BLOCKED: {load_error}",
            "hf_authenticated": False,
            "repo_access": {},
            "hardware": _empty_hardware(),
            "cloud_readiness": _empty_cloud_readiness(),
            "errors": errors,
            "warnings": [],
            "paths": {},
            "skypilot_compat_issue": False,
        }
    assert cfg is not None
    hf_authenticated = False
    hf_token: bool | None = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            future = executor.submit(whoami)
            future.result(timeout=HF_API_TIMEOUT_SECONDS)
            hf_authenticated = True
            hf_token = True
        except HFValidationError:
            errors.append("Invalid HF token")
        except TimeoutError:
            warnings.append(f"HF auth check timed out after {HF_API_TIMEOUT_SECONDS}s")
            hf_token = None
        except Exception as e:
            warnings.append(f"HF auth check failed (may be transient): {e}")
            hf_token = None

    for repo_id, repo_types in get_repos(cfg).items():
        if repo_id in repo_access:
            continue
        for repo_type in repo_types:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        auth_check, repo_id, repo_type=repo_type, token=hf_token
                    )
                    future.result(timeout=HF_API_TIMEOUT_SECONDS)
                repo_access[repo_id] = "valid"
            except GatedRepoError:
                repo_access[repo_id] = "gated"
                errors.append(f"Gated {repo_type} requires access grant: {repo_id}")
            except RepositoryNotFoundError:
                repo_access[repo_id] = "not_found"
                errors.append(f"{repo_type.title()} not found: {repo_id}")
            except TimeoutError:
                repo_access[repo_id] = "error"
                warnings.append(
                    f"Timed out checking {repo_id} after {HF_API_TIMEOUT_SECONDS}s"
                )
            except HfHubHTTPError as e:
                repo_access[repo_id] = "error"
                errors.append(f"HF Hub error for {repo_id}: {str(e)}")
            except Exception as e:
                repo_access[repo_id] = "error"
                errors.append(f"Error checking {repo_id}: {str(e)}")
            break

    hw_errors, hw_warnings, hardware = check_hardware(cfg)
    errors.extend(hw_errors)
    warnings.extend(hw_warnings)

    target_cloud = cloud if cloud and cloud != "local" else ""
    cloud_errors, cloud_warnings, cloud_readiness = check_cloud_readiness(
        target_cloud=target_cloud,
    )
    errors.extend(cloud_errors)
    warnings.extend(cloud_warnings)

    has_compat_issue = any(
        "SkyPilot API compatibility" in msg for msg in [*cloud_errors, *cloud_warnings]
    )

    dataset_checks = validate_datasets(cfg, client_cwd=client_cwd)
    for ds_key, ds_status in dataset_checks.items():
        if ds_status == "not_found":
            errors.append(
                f"Dataset '{ds_key}' is not a registered Oumi dataset, not found on "
                "HuggingFace Hub, and no local dataset_path provided. Use a full HF ID "
                "(e.g., 'yahma/alpaca-cleaned'), a registered name (e.g., "
                "'text_sft_jsonl'), or set dataset_path."
            )
        elif ds_status == "warning_timeout":
            warnings.append(f"HF Hub probe for dataset '{ds_key}' timed out (5s)")

    if target_cloud:
        path_results = validate_paths_cloud(cfg, config_path, client_cwd, target_cloud)
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

    is_blocking = len(errors) > 0
    if is_blocking:
        summary = (
            f"BLOCKED: {len(errors)} issue(s) must be resolved before running. "
            f"First: {errors[0]}"
        )
    elif warnings:
        summary = (
            f"Ready with {len(warnings)} warning(s) (may be fine for remote clusters)"
        )
    else:
        summary = "Ready: all checks passed"

    result: PreFlightCheckResponse = {
        "blocking": is_blocking,
        "summary": summary,
        "hf_authenticated": hf_authenticated,
        "repo_access": repo_access,
        "hardware": hardware,
        "cloud_readiness": cloud_readiness,
        "errors": errors,
        "warnings": warnings,
        "paths": path_results,
        "skypilot_compat_issue": has_compat_issue,
    }

    if dataset_checks:
        result["dataset_checks"] = dataset_checks

    if target_cloud:
        all_cfgs = get_all_configs()
        task_type = cfg.get("task_type", "") or ""
        if not task_type:
            if cfg.get("training"):
                task_type = "sft"
            elif cfg.get("evaluation") or cfg.get("tasks"):
                task_type = "eval"
            elif cfg.get("generation") and cfg.get("input_path"):
                task_type = "infer"
        suggested = search_configs_service(
            all_cfgs, query=[task_type] or ["sft"], limit=5
        )
        result["suggested_configs"] = [c["path"] for c in suggested]

    return result
