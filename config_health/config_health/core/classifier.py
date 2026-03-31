"""Config classification — detects type, extracts metadata, infers GPU tier."""

from __future__ import annotations

import os

from config_health.core.models import (
    REMOTE_ENGINES,
    ConfigEntry,
    ConfigType,
    GpuTier,
)
from config_health.core.scanner import get_category, get_model_family, load_yaml_cached

# Keys/paths that strongly signal a config type
_PATH_TYPE_HINTS: list[tuple[str, ConfigType]] = [
    ("/sft/", ConfigType.TRAINING),
    ("/dpo/", ConfigType.TRAINING),
    ("/grpo/", ConfigType.TRAINING),
    ("/gkd/", ConfigType.TRAINING),
    ("/gold/", ConfigType.TRAINING),
    ("/evaluation/", ConfigType.EVALUATION),
    ("/inference/", ConfigType.INFERENCE),
    ("/judges/", ConfigType.JUDGE),
    ("/synthesis/", ConfigType.SYNTHESIS),
    ("/quantization/", ConfigType.QUANTIZATION),
    ("/analyze/", ConfigType.ANALYZE),
    ("/deploy/", ConfigType.JOB),
]

_FILENAME_TYPE_HINTS: list[tuple[str, ConfigType]] = [
    ("train", ConfigType.TRAINING),
    ("dpo", ConfigType.TRAINING),
    ("sft", ConfigType.TRAINING),
    ("grpo", ConfigType.TRAINING),
    ("eval", ConfigType.EVALUATION),
    ("infer", ConfigType.INFERENCE),
    ("job", ConfigType.JOB),
    ("judge", ConfigType.JUDGE),
    ("synth", ConfigType.SYNTHESIS),
    ("quant", ConfigType.QUANTIZATION),
    ("analyz", ConfigType.ANALYZE),
    ("tune", ConfigType.TUNING),
    ("deploy", ConfigType.JOB),
]

# Use the shared REMOTE_ENGINES constant for API engine detection
_API_ENGINES = REMOTE_ENGINES


def classify_config(yaml_path: str, repo_root: str) -> ConfigEntry:
    """Classify a single YAML config file."""
    rel_path = os.path.relpath(yaml_path, repo_root)
    category = get_category(yaml_path, repo_root)
    model_family = get_model_family(yaml_path, repo_root)

    # Load raw YAML
    raw_data = _load_yaml_safe(yaml_path)
    if raw_data is None:
        return ConfigEntry(
            path=rel_path,
            abs_path=yaml_path,
            category=category,
            model_family=model_family,
            parse_error="Failed to load YAML",
        )

    # Detect type from content + path
    config_type = _detect_type(raw_data, yaml_path)

    # Validate by parsing with the detected oumi config class
    config_class_name, parse_error = _validate_with_oumi(yaml_path, config_type)

    # Extract metadata
    model_name = _extract_model_name(raw_data)
    datasets = _extract_datasets(raw_data)
    engine = _extract_engine(raw_data)
    gpu_tier = _infer_gpu_tier(raw_data, config_type, yaml_path)

    return ConfigEntry(
        path=rel_path,
        abs_path=yaml_path,
        config_type=config_type,
        config_class_name=config_class_name,
        model_name=model_name,
        model_family=model_family,
        gpu_tier=gpu_tier,
        category=category,
        datasets=datasets,
        engine=engine,
        parse_error=parse_error,
    )


def _load_yaml_safe(yaml_path: str) -> dict | None:
    """Load YAML without OmegaConf interpolation. Uses shared cache."""
    return load_yaml_cached(yaml_path)


def _detect_type(data: dict, filepath: str) -> ConfigType:
    """Detect config type from YAML keys and file path."""
    keys = set(data.keys())
    basename = os.path.basename(filepath).lower()
    path_lower = filepath.lower()

    # Job configs have very distinctive structure
    if "resources" in keys and ("run" in keys or "setup" in keys):
        return ConfigType.JOB

    # Judge configs
    if "judge_model" in keys or "/judges/" in path_lower:
        return ConfigType.JUDGE

    # Synthesis configs
    if "synthesis" in keys:
        return ConfigType.SYNTHESIS

    # Quantization configs
    if "quantization" in keys and "training" not in keys:
        return ConfigType.QUANTIZATION

    # Analyze configs
    if "analyzers" in keys or "dataset_source" in keys:
        return ConfigType.ANALYZE

    # Async evaluation configs — wrap an evaluation config inside `evaluation:` key
    if "evaluation" in keys and (
        "checkpoints_dir" in keys or "polling_interval" in keys
    ):
        return ConfigType.ASYNC_EVALUATION

    # Tuning configs
    if "tuning" in keys:
        return ConfigType.TUNING

    # Training configs — have training section
    if "training" in keys and "data" in keys:
        return ConfigType.TRAINING

    # Evaluation configs — have tasks section
    if "tasks" in keys:
        if "async" in basename:
            return ConfigType.ASYNC_EVALUATION
        return ConfigType.EVALUATION

    # Use path hints
    for hint, ctype in _PATH_TYPE_HINTS:
        if hint in path_lower:
            return ctype

    # Use filename hints
    for hint, ctype in _FILENAME_TYPE_HINTS:
        if hint in basename:
            return ctype

    # Inference — has engine but not training/tasks
    if "engine" in keys or "inference_engine" in keys:
        return ConfigType.INFERENCE

    return ConfigType.UNKNOWN


def _validate_with_oumi(
    yaml_path: str, detected_type: ConfigType
) -> tuple[str, str | None]:
    """Try to parse with the oumi config class for the detected type.

    Returns (class_name, error_or_None).
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

    all_classes = [
        TrainingConfig,
        InferenceConfig,
        EvaluationConfig,
        JobConfig,
        JudgeConfig,
        SynthesisConfig,
        QuantizationConfig,
        AnalyzeConfig,
        AsyncEvaluationConfig,
        TuningConfig,
    ]

    # Try the detected type's class first
    primary_cls = type_to_class.get(detected_type)
    if primary_cls:
        try:
            primary_cls.from_yaml(yaml_path)
            return primary_cls.__name__, None
        except Exception:
            pass

    # Fall back: try all classes
    for cls in all_classes:
        try:
            cls.from_yaml(yaml_path)
            return cls.__name__, None
        except Exception:
            continue

    return "Unknown", "Could not parse with any oumi config class"


def _extract_model_name(data: dict) -> str | None:
    """Extract model_name from YAML data."""
    model = data.get("model", {})
    if isinstance(model, dict):
        return model.get("model_name")
    return None


def _extract_datasets(data: dict) -> list[str]:
    """Extract dataset names from YAML data."""
    datasets: list[str] = []
    data_section = data.get("data", {})
    if not isinstance(data_section, dict):
        return datasets
    for split in ("train", "evaluation", "test"):
        split_data = data_section.get(split, {})
        if isinstance(split_data, dict):
            ds_list = split_data.get("datasets", [])
            if isinstance(ds_list, list):
                for ds in ds_list:
                    if isinstance(ds, dict) and "dataset_name" in ds:
                        datasets.append(ds["dataset_name"])
    return datasets


def _extract_engine(data: dict) -> str | None:
    """Extract inference engine from YAML data."""
    # Top-level engine field (inference configs)
    engine = data.get("engine") or data.get("inference_engine")
    if isinstance(engine, str):
        return engine.upper()
    # Judge configs nest engine inside judge_model or model
    for section_key in ("judge_model", "model"):
        section = data.get(section_key, {})
        if isinstance(section, dict):
            eng = section.get("engine") or section.get("inference_engine")
            if isinstance(eng, str):
                return eng.upper()
    return None


def _infer_gpu_tier(data: dict, config_type: ConfigType, filepath: str) -> GpuTier:
    """Infer GPU tier from config content."""
    basename = os.path.basename(filepath).lower()

    # Filename overrides
    if "macos" in basename or "mps" in basename:
        return GpuTier.CPU
    if "multi_gpu" in basename or "multi-gpu" in basename:
        return GpuTier.MULTI_GPU

    # API inference/eval configs
    if config_type in (ConfigType.INFERENCE, ConfigType.EVALUATION):
        engine = data.get("engine", data.get("inference_engine", ""))
        if isinstance(engine, str) and engine.upper() in _API_ENGINES:
            return GpuTier.CPU

    # Job configs — check accelerator count
    resources = data.get("resources", {})
    if isinstance(resources, dict):
        accel = resources.get("accelerators", "")
        if isinstance(accel, str) and ":" in accel:
            try:
                count = int(accel.split(":")[-1])
                return GpuTier.MULTI_GPU if count > 1 else GpuTier.SINGLE_GPU
            except ValueError:
                pass
        num_nodes = data.get("num_nodes", 1)
        if isinstance(num_nodes, int) and num_nodes > 1:
            return GpuTier.MULTI_GPU

    # FSDP
    fsdp = data.get("fsdp", {})
    if isinstance(fsdp, dict) and fsdp.get("enable_fsdp"):
        return GpuTier.MULTI_GPU

    # DeepSpeed
    deepspeed = data.get("deepspeed", {})
    if isinstance(deepspeed, dict) and deepspeed.get("enable_deepspeed"):
        return GpuTier.MULTI_GPU

    # Training/tuning generally need GPU
    if config_type in (ConfigType.TRAINING, ConfigType.TUNING):
        return GpuTier.SINGLE_GPU

    # Judge, synthesis, analyze — usually CPU
    if config_type in (ConfigType.JUDGE, ConfigType.SYNTHESIS, ConfigType.ANALYZE):
        return GpuTier.CPU

    # Eval with local model needs GPU
    if config_type == ConfigType.EVALUATION:
        model = data.get("model", {})
        if isinstance(model, dict) and model.get("model_name"):
            return GpuTier.SINGLE_GPU

    # Inference with local model
    if config_type == ConfigType.INFERENCE:
        model = data.get("model", {})
        if isinstance(model, dict) and model.get("model_name"):
            return GpuTier.SINGLE_GPU

    return GpuTier.CPU
