"""CI entry point — run static config health checks.

Uses the same _build_report pipeline as the CLI to avoid divergence.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Run config health checks for CI.

    Exit codes:
      0 = all checks pass
      1 = failures found
      2 = infrastructure error (can't find repo root)
    """
    from config_health.cli import _build_report
    from config_health.core.models import CheckStatus
    from config_health.core.scanner import find_repo_root

    try:
        repo_root = find_repo_root()
    except FileNotFoundError:
        print("ERROR: Could not find oumi repo root", file=sys.stderr)
        sys.exit(2)

    print("Running config health checks...")
    report = _build_report(
        repo_root,
        hub_check=False,
        quick=False,
    )

    failures = [r for r in report.check_results if r.status == CheckStatus.FAIL]
    warnings = [r for r in report.check_results if r.status == CheckStatus.WARN]

    print(
        f"\nResults: {len(report.entries)} configs, {len(failures)} failures, {len(warnings)} warnings"
    )

    if failures:
        print("\nFailures:")
        for r in failures:
            print(f"  FAIL: {r.config_path}: {r.message}")
        sys.exit(1)

    print("\nAll config checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
