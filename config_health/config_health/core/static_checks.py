"""Static health checks — no GPU, no network needed."""

from __future__ import annotations

import os

import yaml

from config_health.core.models import (
    CheckResult,
    CheckStatus,
    ConfigEntry,
    ConfigType,
    Severity,
)


def run_static_checks(entry: ConfigEntry, repo_root: str) -> list[CheckResult]:
    """Run all static checks on a config entry."""
    results: list[CheckResult] = []
    results.append(_check_parse(entry))

    # Run finalize_and_validate (catches deeper oumi validation)
    if not entry.parse_error:
        results.append(_check_finalize_and_validate(entry))

    # Load raw data for deeper checks
    data = _load_yaml(entry.abs_path)
    if data is None:
        return results

    results.extend(_check_cross_field(entry, data))
    results.extend(_check_file_references(entry, data, repo_root))
    results.extend(_check_completeness(entry, data))
    return results


def _check_parse(entry: ConfigEntry) -> CheckResult:
    """Check that the config parses into an oumi config class."""
    if entry.parse_error:
        return CheckResult(
            config_path=entry.path,
            check_name="parse",
            status=CheckStatus.FAIL,
            message=entry.parse_error,
            severity=Severity.ERROR,
        )
    return CheckResult(
        config_path=entry.path,
        check_name="parse",
        status=CheckStatus.PASS,
        message=f"Parses as {entry.config_class_name}",
        severity=Severity.INFO,
    )


def _check_finalize_and_validate(entry: ConfigEntry) -> CheckResult:
    """Run oumi's finalize_and_validate() to catch deeper validation errors.

    This catches issues that __post_init__ doesn't, such as:
    - DataParams requiring at least one training dataset
    - ModelParams adapter auto-detection
    - Collator consistency across splits
    """
    from oumi.core.configs import (
        AnalyzeConfig,
        AsyncEvaluationConfig,
        EvaluationConfig,
        InferenceConfig,
        JobConfig,
        JudgeConfig,
        QuantizationConfig,
        SynthesisConfig,
        TrainingConfig,
        TuningConfig,
    )

    type_to_class = {
        ConfigType.TRAINING: TrainingConfig,
        ConfigType.INFERENCE: InferenceConfig,
        ConfigType.EVALUATION: EvaluationConfig,
        ConfigType.JOB: JobConfig,
        ConfigType.JUDGE: JudgeConfig,
        ConfigType.SYNTHESIS: SynthesisConfig,
        ConfigType.QUANTIZATION: QuantizationConfig,
        ConfigType.ANALYZE: AnalyzeConfig,
        ConfigType.ASYNC_EVALUATION: AsyncEvaluationConfig,
        ConfigType.TUNING: TuningConfig,
    }

    cls = type_to_class.get(entry.config_type)
    if not cls:
        return CheckResult(
            config_path=entry.path,
            check_name="finalize_validate",
            status=CheckStatus.SKIP,
            message="Unknown config type",
            severity=Severity.INFO,
        )

    try:
        config = cls.from_yaml(entry.abs_path)

        # Training configs often omit datasets (they're templates).
        # Inject a dummy dataset so finalize_and_validate doesn't fail on that.
        if isinstance(config, TrainingConfig) and len(config.data.train.datasets) == 0:
            from oumi.core.configs.params.data_params import DatasetParams

            config.data.train.datasets.append(
                DatasetParams(dataset_name="__health_check_dummy__")
            )

        config.finalize_and_validate()
        return CheckResult(
            config_path=entry.path,
            check_name="finalize_validate",
            status=CheckStatus.PASS,
            message="finalize_and_validate passed",
            severity=Severity.INFO,
        )
    except Exception as e:
        msg = str(e)
        # Ignore HardwareExceptions (e.g., CUDA not available)
        if "HardwareException" in type(e).__name__:
            return CheckResult(
                config_path=entry.path,
                check_name="finalize_validate",
                status=CheckStatus.SKIP,
                message=f"Hardware requirement: {msg[:120]}",
                severity=Severity.INFO,
            )
        return CheckResult(
            config_path=entry.path,
            check_name="finalize_validate",
            status=CheckStatus.FAIL,
            message=f"finalize_and_validate failed: {msg[:200]}",
            severity=Severity.ERROR,
            details=msg[:500] if len(msg) > 200 else None,
        )


def _check_cross_field(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Cross-field validation checks."""
    results: list[CheckResult] = []

    if entry.config_type == ConfigType.TRAINING:
        results.extend(_check_training_fields(entry, data))
    elif entry.config_type == ConfigType.INFERENCE:
        results.extend(_check_inference_fields(entry, data))
    elif entry.config_type == ConfigType.EVALUATION:
        results.extend(_check_evaluation_fields(entry, data))

    return results


def _check_training_fields(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Training-specific cross-field checks."""
    results: list[CheckResult] = []
    training = data.get("training", {})
    if not isinstance(training, dict):
        return results

    # Check output_dir is set
    if not training.get("output_dir"):
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="training_output_dir",
                status=CheckStatus.WARN,
                message="No output_dir set — training results won't be saved",
                severity=Severity.WARNING,
            )
        )

    # Check FSDP transformer_layer_cls
    fsdp = data.get("fsdp", {})
    if isinstance(fsdp, dict) and fsdp.get("enable_fsdp"):
        if not fsdp.get("transformer_layer_cls"):
            auto_wrap = fsdp.get("auto_wrap_policy", "")
            if isinstance(auto_wrap, str) and "TRANSFORMER" in auto_wrap.upper():
                results.append(
                    CheckResult(
                        config_path=entry.path,
                        check_name="fsdp_layer_cls",
                        status=CheckStatus.WARN,
                        message="FSDP TRANSFORMER_BASED_WRAP without transformer_layer_cls",
                        severity=Severity.WARNING,
                    )
                )

    # Check LoRA targets when peft is enabled
    peft = data.get("peft", {})
    if isinstance(peft, dict) and training.get("use_peft"):
        if not peft.get("lora_target_modules"):
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="lora_targets",
                    status=CheckStatus.WARN,
                    message="PEFT enabled but no lora_target_modules specified",
                    severity=Severity.WARNING,
                )
            )

    # Check data section has datasets
    data_section = data.get("data", {})
    if isinstance(data_section, dict):
        train_data = data_section.get("train", {})
        if isinstance(train_data, dict):
            datasets = train_data.get("datasets", [])
            if not datasets:
                results.append(
                    CheckResult(
                        config_path=entry.path,
                        check_name="training_dataset",
                        status=CheckStatus.WARN,
                        message="No training datasets specified",
                        severity=Severity.WARNING,
                    )
                )

    return results


def _check_inference_fields(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Inference-specific checks."""
    results: list[CheckResult] = []
    gen = data.get("generation", {})
    if isinstance(gen, dict):
        if not gen.get("max_new_tokens"):
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="inference_max_tokens",
                    status=CheckStatus.WARN,
                    message="No max_new_tokens set — generation may run indefinitely",
                    severity=Severity.WARNING,
                )
            )
    return results


def _check_evaluation_fields(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Evaluation-specific checks."""
    results: list[CheckResult] = []
    tasks = data.get("tasks", [])
    if isinstance(tasks, list) and len(tasks) == 0:
        results.append(
            CheckResult(
                config_path=entry.path,
                check_name="eval_tasks",
                status=CheckStatus.WARN,
                message="No evaluation tasks defined",
                severity=Severity.WARNING,
            )
        )
    return results


def _check_file_references(
    entry: ConfigEntry, data: dict, repo_root: str
) -> list[CheckResult]:
    """Check that referenced files exist."""
    results: list[CheckResult] = []

    # Check for deepspeed config file references
    ds = data.get("deepspeed", {})
    if isinstance(ds, dict):
        config_file = ds.get("config_file")
        if config_file and isinstance(config_file, str):
            full_path = (
                config_file
                if os.path.isabs(config_file)
                else os.path.join(repo_root, config_file)
            )
            if not os.path.exists(full_path):
                results.append(
                    CheckResult(
                        config_path=entry.path,
                        check_name="file_ref_deepspeed",
                        status=CheckStatus.FAIL,
                        message=f"DeepSpeed config not found: {config_file}",
                        severity=Severity.ERROR,
                    )
                )

    # Check for job config referencing training config via oumi CLI
    if entry.config_type == ConfigType.JOB:
        run_cmd = data.get("run", "")
        if isinstance(run_cmd, str):
            import re

            # Match "oumi <command> ... -c <config_path>" patterns
            for match in re.finditer(
                r"oumi\s+(?:train|evaluate|infer|distributed\s+\S+\s+-m\s+oumi\s+\S+)"
                r".*?-c\s+(\S+)",
                run_cmd,
            ):
                ref_path = match.group(1).strip("'\"\\")
                if ref_path.startswith("oumi://"):
                    ref_path = ref_path[len("oumi://") :]
                # Skip variable references like $VAR
                if ref_path.startswith("$"):
                    continue
                full_ref = os.path.join(repo_root, ref_path)
                if not os.path.exists(full_ref):
                    results.append(
                        CheckResult(
                            config_path=entry.path,
                            check_name="file_ref_job_config",
                            status=CheckStatus.FAIL,
                            message=f"Referenced config not found: {ref_path}",
                            severity=Severity.ERROR,
                        )
                    )

    return results


def _check_completeness(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Check for missing recommended fields."""
    results: list[CheckResult] = []

    # Model configs should have model_name
    if entry.config_type in (
        ConfigType.TRAINING,
        ConfigType.INFERENCE,
        ConfigType.EVALUATION,
    ):
        model = data.get("model", {})
        if isinstance(model, dict) and not model.get("model_name"):
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="model_name",
                    status=CheckStatus.WARN,
                    message="No model_name specified",
                    severity=Severity.WARNING,
                )
            )

    return results


def _load_yaml(path: str) -> dict | None:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None
