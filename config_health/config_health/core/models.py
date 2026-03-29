"""Data models for config health checking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class GpuTier(int, Enum):
    CPU = 0
    SINGLE_GPU = 1
    MULTI_GPU = 2

    @property
    def label(self) -> str:
        return {0: "CPU/MPS", 1: "Single GPU", 2: "Multi-GPU"}[self.value]

    @property
    def badge_color(self) -> str:
        return {0: "gray", 1: "blue", 2: "purple"}[self.value]


class ConfigType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    JOB = "job"
    JUDGE = "judge"
    SYNTHESIS = "synthesis"
    QUANTIZATION = "quantization"
    ANALYZE = "analyze"
    ASYNC_EVALUATION = "async_evaluation"
    TUNING = "tuning"
    UNKNOWN = "unknown"

    @property
    def badge_color(self) -> str:
        colors = {
            "training": "indigo",
            "inference": "green",
            "evaluation": "amber",
            "job": "slate",
            "judge": "rose",
            "synthesis": "cyan",
            "quantization": "orange",
            "analyze": "teal",
            "async_evaluation": "yellow",
            "tuning": "violet",
            "unknown": "gray",
        }
        return colors.get(self.value, "gray")


# Mapping from config class name to ConfigType
# Remote inference engines whose model_name is a provider-specific identifier,
# not a HuggingFace Hub repo ID. Single source of truth — import this constant
# instead of duplicating the set in each module.
REMOTE_ENGINES = frozenset(
    {
        "ANTHROPIC",
        "OPENAI",
        "GOOGLE",
        "GOOGLE_GEMINI",
        "GOOGLE_VERTEX",
        "OPENROUTER",
        "TOGETHER",
        "FIREWORKS",
        "PARASAIL",
        "LAMBDA",
        "REMOTE",
        "REMOTE_VLLM",
    }
)


CONFIG_CLASS_TO_TYPE: dict[str, ConfigType] = {
    "TrainingConfig": ConfigType.TRAINING,
    "InferenceConfig": ConfigType.INFERENCE,
    "EvaluationConfig": ConfigType.EVALUATION,
    "JobConfig": ConfigType.JOB,
    "JudgeConfig": ConfigType.JUDGE,
    "SynthesisConfig": ConfigType.SYNTHESIS,
    "QuantizationConfig": ConfigType.QUANTIZATION,
    "AnalyzeConfig": ConfigType.ANALYZE,
    "AsyncEvaluationConfig": ConfigType.ASYNC_EVALUATION,
    "TuningConfig": ConfigType.TUNING,
}


@dataclass
class ConfigEntry:
    """A discovered config file with its metadata."""

    path: str  # relative to repo root
    abs_path: str
    config_type: ConfigType = ConfigType.UNKNOWN
    config_class_name: str = "Unknown"
    model_name: str | None = None
    model_family: str | None = None
    gpu_tier: GpuTier = GpuTier.CPU
    category: str = "unknown"  # recipes, apis, examples, projects
    datasets: list[str] = field(default_factory=list)
    engine: str | None = None  # inference engine (e.g. NATIVE, OPENAI, ANTHROPIC)
    parse_error: str | None = None

    @property
    def status(self) -> CheckStatus:
        """Overall status based on parse_error."""
        if self.parse_error:
            return CheckStatus.FAIL
        if self.config_type == ConfigType.UNKNOWN:
            return CheckStatus.WARN
        return CheckStatus.PASS

    @property
    def short_path(self) -> str:
        """Path without configs/ prefix."""
        if self.path.startswith("configs/"):
            return self.path[len("configs/") :]
        return self.path


@dataclass
class CheckResult:
    """Result of a single health check on a config."""

    config_path: str
    check_name: str
    status: CheckStatus
    message: str
    severity: Severity
    details: str | None = None
    duration_s: float | None = None


@dataclass
class CoverageGap:
    """A missing config type for a model family."""

    model_family: str
    missing_types: list[str]
    existing_types: list[str]
    category: str = "recipes"

    @property
    def completeness(self) -> float:
        total = len(self.missing_types) + len(self.existing_types)
        return len(self.existing_types) / total if total else 0.0


@dataclass
class ArchCoverageEntry:
    """Coverage info for an HF model architecture (model_type)."""

    model_type: str  # e.g. "llama", "qwen2", "gemma2"
    model_class: str  # e.g. "LlamaForCausalLM"
    config_types: list[str] = field(default_factory=list)  # e.g. ["training", "inference"]
    model_names: list[str] = field(default_factory=list)  # HF repo IDs using this arch
    config_count: int = 0
    is_vlm: bool = False  # vision-language model
    in_oumi_registry: bool = False  # registered in Oumi's supported_models
    oumi_tested: bool = False  # marked as tested in Oumi's registry


@dataclass
class OptimizationSuggestion:
    """A suggested improvement for a config."""

    config_path: str
    category: str  # performance, compatibility, modernization, best_practice
    title: str
    suggestion: str
    priority: str = "medium"  # low, medium, high


@dataclass
class HealthReport:
    """Full health report for all configs."""

    entries: list[ConfigEntry] = field(default_factory=list)
    check_results: list[CheckResult] = field(default_factory=list)
    coverage_gaps: list[CoverageGap] = field(default_factory=list)
    suggestions: list[OptimizationSuggestion] = field(default_factory=list)
    vram_estimates: dict[str, dict] = field(default_factory=dict)  # path -> estimate dict
    dry_run_results: dict[str, dict] = field(default_factory=dict)  # path -> result dict
    phase_durations_s: dict[str, float] = field(default_factory=dict)  # phase -> seconds
    environment: dict[str, str] = field(default_factory=dict)  # runtime env info
    scan_duration_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.entries)

    @property
    def fail_paths(self) -> set[str]:
        """Configs with at least one FAIL result."""
        stats = self._status_stats()
        return stats[0]

    @property
    def warn_paths(self) -> set[str]:
        """Configs with at least one WARN but no FAIL results."""
        stats = self._status_stats()
        return stats[1]

    @property
    def pass_count(self) -> int:
        """Configs with no FAIL and no WARN results."""
        fail, warn = self._status_stats()
        return len(self.entries) - len(fail) - len(warn)

    @property
    def fail_count(self) -> int:
        return len(self.fail_paths)

    @property
    def warn_count(self) -> int:
        return len(self.warn_paths)

    def _status_stats(self) -> tuple[set[str], set[str]]:
        """Compute fail_paths and warn_paths in a single pass. Cached."""
        # Use a simple instance-level cache keyed on check_results length
        # to avoid recomputing on repeated property access within the same state.
        cache_key = len(self.check_results)
        cached = getattr(self, "_status_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]
        fail = {r.config_path for r in self.check_results if r.status == CheckStatus.FAIL}
        warn = {r.config_path for r in self.check_results if r.status == CheckStatus.WARN} - fail
        self._status_cache = (cache_key, fail, warn)  # type: ignore[attr-defined]
        return fail, warn

    def results_for(self, config_path: str) -> list[CheckResult]:
        return [r for r in self.check_results if r.config_path == config_path]

    def suggestions_for(self, config_path: str) -> list[OptimizationSuggestion]:
        return [s for s in self.suggestions if s.config_path == config_path]

    def entries_by_type(self) -> dict[ConfigType, list[ConfigEntry]]:
        result: dict[ConfigType, list[ConfigEntry]] = {}
        for entry in self.entries:
            result.setdefault(entry.config_type, []).append(entry)
        return result

    def entries_by_family(self) -> dict[str, list[ConfigEntry]]:
        result: dict[str, list[ConfigEntry]] = {}
        for entry in self.entries:
            key = entry.model_family or "other"
            result.setdefault(key, []).append(entry)
        return result

    def to_json(self, path: str | Path) -> None:
        """Save report as JSON."""

        def _serialize(obj: object) -> object:
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Cannot serialize {type(obj)}")

        data = {
            "entries": [asdict(e) for e in self.entries],
            "check_results": [asdict(r) for r in self.check_results],
            "coverage_gaps": [asdict(g) for g in self.coverage_gaps],
            "suggestions": [asdict(s) for s in self.suggestions],
            "vram_estimates": self.vram_estimates,
            "dry_run_results": self.dry_run_results,
            "phase_durations_s": self.phase_durations_s,
            "environment": self.environment,
            "scan_duration_s": self.scan_duration_s,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_serialize)

    @classmethod
    def from_json(cls, path: str | Path) -> "HealthReport":
        """Load report from JSON."""
        with open(path) as f:
            data = json.load(f)

        report = cls()
        for e in data.get("entries", []):
            e["config_type"] = ConfigType(e["config_type"])
            e["gpu_tier"] = GpuTier(e["gpu_tier"])
            report.entries.append(ConfigEntry(**e))
        for r in data.get("check_results", []):
            r["status"] = CheckStatus(r["status"])
            r["severity"] = Severity(r["severity"])
            report.check_results.append(CheckResult(**r))
        for g in data.get("coverage_gaps", []):
            report.coverage_gaps.append(CoverageGap(**g))
        for s in data.get("suggestions", []):
            report.suggestions.append(OptimizationSuggestion(**s))
        report.vram_estimates = data.get("vram_estimates", {})
        report.dry_run_results = data.get("dry_run_results", {})
        report.phase_durations_s = data.get("phase_durations_s", {})
        report.environment = data.get("environment", {})
        report.scan_duration_s = data.get("scan_duration_s", 0.0)
        return report
