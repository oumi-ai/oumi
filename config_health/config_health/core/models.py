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
    def pass_count(self) -> int:
        return self._count_status(CheckStatus.PASS)

    @property
    def fail_count(self) -> int:
        return self._count_status(CheckStatus.FAIL)

    @property
    def warn_count(self) -> int:
        return self._count_status(CheckStatus.WARN)

    def _count_status(self, status: CheckStatus) -> int:
        return sum(1 for r in self.check_results if r.status == status)

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
