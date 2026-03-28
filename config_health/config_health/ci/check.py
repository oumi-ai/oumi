"""CI entry point — run static config health checks."""

from __future__ import annotations

import sys


def main() -> None:
    """Run config health checks for CI.

    Exit codes:
      0 = all checks pass
      1 = failures found
    """
    from config_health.core.classifier import classify_config
    from config_health.core.models import CheckStatus, HealthReport
    from config_health.core.scanner import find_repo_root, scan_config_paths
    from config_health.core.static_checks import run_static_checks

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        print("ERROR: Could not find oumi repo root", file=sys.stderr)
        sys.exit(2)

    report = HealthReport()
    paths = scan_config_paths(repo_root)
    print(f"Scanning {len(paths)} configs...")

    for p in paths:
        try:
            entry = classify_config(p, repo_root)
        except Exception as exc:
            import os

            from config_health.core.models import ConfigEntry

            entry = ConfigEntry(
                path=os.path.relpath(p, repo_root),
                abs_path=p,
                parse_error=f"Classification crashed: {exc}",
            )
        report.entries.append(entry)
        try:
            results = run_static_checks(entry, repo_root)
        except Exception as exc:
            from config_health.core.models import CheckResult, Severity

            results = [CheckResult(
                config_path=entry.path,
                check_name="static_checks",
                status=CheckStatus.FAIL,
                message=f"Static checks crashed: {exc}",
                severity=Severity.ERROR,
            )]
        report.check_results.extend(results)

    failures = [r for r in report.check_results if r.status == CheckStatus.FAIL]
    warnings = [r for r in report.check_results if r.status == CheckStatus.WARN]

    print(f"\nResults: {len(report.entries)} configs, {len(failures)} failures, {len(warnings)} warnings")

    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  FAIL: {r.config_path}: {r.message}")
        sys.exit(1)

    print("\nAll config checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
