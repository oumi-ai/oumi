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


def run_static_checks(
    entry: ConfigEntry, repo_root: str, *, skip_finalize: bool = False
) -> list[CheckResult]:
    """Run all static checks on a config entry."""
    results: list[CheckResult] = []
    results.append(_check_parse(entry))

    # Run finalize_and_validate (catches deeper oumi validation)
    # --quick mode skips this (~3s per config for the oumi import)
    if not entry.parse_error and not skip_finalize:
        results.append(_check_finalize_and_validate(entry))

    # Load raw data for deeper checks
    data = _load_yaml(entry.abs_path)
    if data is None:
        return results

    # Skip deeper checks for configs that failed to parse — they'll have
    # noisy false positives (e.g., accelerate.yaml flagged for missing model_name)
    if entry.parse_error:
        return results

    results.extend(_check_unknown_keys(entry, data))
    results.extend(_check_cross_field(entry, data))
    results.extend(_check_trainer_specific(entry, data))
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


def _check_unknown_keys(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Detect unknown/misspelled YAML keys by comparing against OmegaConf schema.

    Top-level unknown keys → FAIL (almost certainly a typo).
    Nested unknown keys → WARN (could be intentional pass-through like model_kwargs).
    """
    results: list[CheckResult] = []

    # Skip for configs that already failed to parse or have unknown type
    if entry.parse_error or entry.config_type == ConfigType.UNKNOWN:
        return results

    schema = _get_config_schema(entry.config_type)
    if not schema:
        return results

    # Check top-level keys
    for key in data:
        if key not in schema:
            suggestion = _suggest_key(key, schema)
            msg = f"Unknown top-level key '{key}'"
            if suggestion:
                msg += f" — did you mean '{suggestion}'?"
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="unknown_key",
                    status=CheckStatus.FAIL,
                    message=msg,
                    severity=Severity.ERROR,
                )
            )

    # Check nested keys in known sections
    _NESTED_SCHEMAS = _get_nested_schemas()
    for section_key, section_fields in _NESTED_SCHEMAS.items():
        section_data = data.get(section_key)
        if not isinstance(section_data, dict):
            continue
        for key in section_data:
            if key not in section_fields:
                suggestion = _suggest_key(key, section_fields)
                msg = f"Unknown key '{section_key}.{key}'"
                if suggestion:
                    msg += f" — did you mean '{suggestion}'?"
                results.append(
                    CheckResult(
                        config_path=entry.path,
                        check_name="unknown_key",
                        status=CheckStatus.WARN,
                        message=msg,
                        severity=Severity.WARNING,
                    )
                )

    return results


def _suggest_key(unknown: str, valid_keys: object) -> str | None:
    """Suggest the closest valid key using edit distance."""
    best = None
    best_dist = float("inf")
    for valid in valid_keys:
        dist = _edit_distance(unknown.lower(), valid.lower())
        if dist < best_dist and dist <= max(2, len(unknown) // 3):
            best = valid
            best_dist = dist
    return best


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j in range(1, len(b) + 1):
        curr = [j] + [0] * len(a)
        for i in range(1, len(a) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[len(a)]


# Schema cache — built once per process
_schema_cache: dict[str, dict[str, set[str]]] = {}


def _get_config_schema(config_type: ConfigType) -> set[str] | None:
    """Get the set of valid top-level YAML keys for a config type."""
    import dataclasses

    if "top_level" in _schema_cache:
        return _schema_cache["top_level"].get(config_type.value)

    # Build schemas from oumi config dataclasses
    try:
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

        type_to_cls = {
            "training": TrainingConfig,
            "inference": InferenceConfig,
            "evaluation": EvaluationConfig,
            "job": JobConfig,
            "judge": JudgeConfig,
            "synthesis": SynthesisConfig,
            "quantization": QuantizationConfig,
            "analyze": AnalyzeConfig,
            "async_evaluation": AsyncEvaluationConfig,
            "tuning": TuningConfig,
        }

        _schema_cache["top_level"] = {}
        for type_name, cls in type_to_cls.items():
            _schema_cache["top_level"][type_name] = {
                f.name for f in dataclasses.fields(cls)
            }
    except Exception:
        _schema_cache["top_level"] = {}

    return _schema_cache["top_level"].get(config_type.value)


def _get_nested_schemas() -> dict[str, set[str]]:
    """Get valid keys for nested config sections (model, training, data, fsdp, peft)."""
    if "nested" in _schema_cache:
        return _schema_cache["nested"]

    import dataclasses

    result: dict[str, set[str]] = {}
    try:
        from oumi.core.configs.params.data_params import DataParams
        from oumi.core.configs.params.fsdp_params import FSDPParams
        from oumi.core.configs.params.model_params import ModelParams
        from oumi.core.configs.params.peft_params import PeftParams
        from oumi.core.configs.params.training_params import TrainingParams

        for name, cls in [
            ("model", ModelParams),
            ("training", TrainingParams),
            ("data", DataParams),
            ("fsdp", FSDPParams),
            ("peft", PeftParams),
        ]:
            result[name] = {f.name for f in dataclasses.fields(cls)}
    except Exception:
        pass

    _schema_cache["nested"] = result
    return result


def _check_trainer_specific(entry: ConfigEntry, data: dict) -> list[CheckResult]:
    """Validate trainer-type-specific required fields.

    Different trainers (GRPO, GKD, GOLD, DPO, KTO) need specific config sections.
    """
    results: list[CheckResult] = []
    if entry.config_type != ConfigType.TRAINING:
        return results

    training = data.get("training", {})
    if not isinstance(training, dict):
        return results

    trainer_type = training.get("trainer_type", "")
    if not isinstance(trainer_type, str):
        return results

    trainer_upper = trainer_type.upper()

    # GRPO trainers need grpo section and reward_functions
    if "GRPO" in trainer_upper:
        grpo = training.get("grpo")
        reward = training.get("reward_functions")
        if not grpo and not reward:
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="trainer_grpo",
                    status=CheckStatus.FAIL,
                    message=(
                        f"Trainer {trainer_type} requires 'training.grpo' section "
                        "and/or 'training.reward_functions'"
                    ),
                    severity=Severity.ERROR,
                )
            )

    # GKD trainer needs gkd section
    if "GKD" in trainer_upper:
        if not training.get("gkd"):
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="trainer_gkd",
                    status=CheckStatus.FAIL,
                    message=f"Trainer {trainer_type} requires 'training.gkd' section",
                    severity=Severity.ERROR,
                )
            )

    # GOLD trainer needs gold section
    if "GOLD" in trainer_upper:
        if not training.get("gold"):
            results.append(
                CheckResult(
                    config_path=entry.path,
                    check_name="trainer_gold",
                    status=CheckStatus.FAIL,
                    message=f"Trainer {trainer_type} requires 'training.gold' section",
                    severity=Severity.ERROR,
                )
            )

    # DPO trainer needs paired data (train split with specific format)
    if "DPO" in trainer_upper or "KTO" in trainer_upper:
        data_section = data.get("data", {})
        if isinstance(data_section, dict):
            train_data = data_section.get("train", {})
            if isinstance(train_data, dict):
                datasets = train_data.get("datasets", [])
                if datasets and isinstance(datasets, list) and isinstance(datasets[0], dict):
                    # DPO datasets should have a specific format
                    ds = datasets[0]
                    if not ds.get("dataset_name"):
                        results.append(
                            CheckResult(
                                config_path=entry.path,
                                check_name="trainer_dpo_data",
                                status=CheckStatus.WARN,
                                message=f"Trainer {trainer_type}: verify dataset provides preference pairs",
                                severity=Severity.WARNING,
                            )
                        )

    return results


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


from config_health.core.scanner import load_yaml_cached as _load_yaml  # shared cache
