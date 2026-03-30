"""Report diff — compare two report.json files and show what changed."""

from __future__ import annotations

from dataclasses import dataclass, field

from config_health.core.models import CheckStatus, HealthReport


@dataclass
class ReportDiff:
    """Differences between two health reports."""

    new_configs: list[str] = field(default_factory=list)
    removed_configs: list[str] = field(default_factory=list)
    new_failures: list[tuple[str, str]] = field(default_factory=list)  # (path, message)
    resolved_failures: list[tuple[str, str]] = field(default_factory=list)
    new_warnings: list[tuple[str, str]] = field(default_factory=list)
    resolved_warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(
            self.new_configs
            or self.removed_configs
            or self.new_failures
            or self.resolved_failures
            or self.new_warnings
            or self.resolved_warnings
        )

    @property
    def is_regression(self) -> bool:
        return bool(self.new_failures)

    def to_markdown(self) -> str:
        lines: list[str] = []
        w = lines.append

        if not self.has_changes:
            w("No changes detected.")
            return "\n".join(lines)

        w("# Config Health Diff\n")

        if self.new_failures:
            w(f"## New Failures ({len(self.new_failures)})\n")
            for path, msg in self.new_failures:
                w(f"- **{path}**: {msg}")
            w("")

        if self.resolved_failures:
            w(f"## Resolved Failures ({len(self.resolved_failures)})\n")
            for path, msg in self.resolved_failures:
                w(f"- ~~{path}~~: {msg}")
            w("")

        if self.new_warnings:
            w(f"## New Warnings ({len(self.new_warnings)})\n")
            for path, msg in self.new_warnings:
                w(f"- **{path}**: {msg}")
            w("")

        if self.resolved_warnings:
            w(f"## Resolved Warnings ({len(self.resolved_warnings)})\n")
            for path, msg in self.resolved_warnings:
                w(f"- ~~{path}~~: {msg}")
            w("")

        if self.new_configs:
            w(f"## New Configs ({len(self.new_configs)})\n")
            for path in self.new_configs:
                w(f"- {path}")
            w("")

        if self.removed_configs:
            w(f"## Removed Configs ({len(self.removed_configs)})\n")
            for path in self.removed_configs:
                w(f"- ~~{path}~~")
            w("")

        # Summary line
        parts = []
        if self.new_failures:
            parts.append(f"+{len(self.new_failures)} failures")
        if self.resolved_failures:
            parts.append(f"-{len(self.resolved_failures)} failures")
        if self.new_warnings:
            parts.append(f"+{len(self.new_warnings)} warnings")
        if self.resolved_warnings:
            parts.append(f"-{len(self.resolved_warnings)} warnings")
        if self.new_configs:
            parts.append(f"+{len(self.new_configs)} configs")
        if self.removed_configs:
            parts.append(f"-{len(self.removed_configs)} configs")
        w(f"**Summary**: {', '.join(parts)}")

        return "\n".join(lines)


def diff_reports(old: HealthReport, new: HealthReport) -> ReportDiff:
    """Compare two health reports and return the diff."""
    result = ReportDiff()

    old_paths = {e.path for e in old.entries}
    new_paths = {e.path for e in new.entries}
    result.new_configs = sorted(new_paths - old_paths)
    result.removed_configs = sorted(old_paths - new_paths)

    # Compare check results
    def _result_set(report: HealthReport, status: CheckStatus) -> set[tuple[str, str]]:
        return {
            (r.config_path, r.message)
            for r in report.check_results
            if r.status == status
        }

    old_fails = _result_set(old, CheckStatus.FAIL)
    new_fails = _result_set(new, CheckStatus.FAIL)
    result.new_failures = sorted(new_fails - old_fails)
    result.resolved_failures = sorted(old_fails - new_fails)

    old_warns = _result_set(old, CheckStatus.WARN)
    new_warns = _result_set(new, CheckStatus.WARN)
    result.new_warnings = sorted(new_warns - old_warns)
    result.resolved_warnings = sorted(old_warns - new_warns)

    return result
