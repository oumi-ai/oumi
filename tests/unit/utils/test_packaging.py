import importlib.metadata
import sys
from pathlib import Path

import pytest
from packaging import version
from packaging.requirements import Requirement
from packaging.version import Version

from oumi.utils.packaging import (
    PackagePrerequisites,
    _package_error_message,
    _package_prerequisites_error_messages,
    check_package_prerequisites,
    verify_trl_vllm_compatibility,
)


@pytest.mark.parametrize(
    "package_name, actual_version, min_version, max_version, expected_message",
    [
        # Happy path tests (package installed, correct version).
        ("my_package", "1.23.5", "1.20.4", None, None),
        ("my_package", "1.23.5", "1.20.5", None, None),
        ("my_package", "1.23.5", None, "1.23.6", None),
        ("my_package", "1.23.5", None, "1.23.5", None),
        ("my_package", "1.23.5", "1.20.4", "1.23.6", None),
        ("my_package", "1.23.5", "1.20.5", "1.23.5", None),
        ("my_package", "1.23.5", None, None, None),
        # Error cases (package not installed).
        ("my_package", None, None, None, "Package `my_package` is not installed."),
        (
            "my_package",
            None,
            "1.20.0",
            None,
            "Package `my_package` is not installed. Please install: version >= 1.20.0.",
        ),
        # Error cases (package version incompatible).
        (
            "my_package",
            "1.23.5",
            "1.23.6",
            None,
            "Package `my_package` version is 1.23.5, which is not compatible. "
            "Please install: version >= 1.23.6.",
        ),
        (
            "my_package",
            "1.23.5",
            "1.23.6",
            "1.23.7",
            "Package `my_package` version is 1.23.5, which is not compatible. "
            "Please install: 1.23.6 <= version <= 1.23.7.",
        ),
    ],
    ids=[
        "happy_path_greater",
        "happy_path_greater_equal",
        "happy_path_less",
        "happy_path_less_equal",
        "happy_path_inbetween_values",
        "happy_path_exact_version",
        "happy_path_no_version_restrictions",
        "error_case_package_not_installed",
        "error_case_package_not_installed_specific_version_required",
        "error_case_package_version_incompatible_too_low_one_boundary",
        "error_case_package_version_incompatible_too_low_two_boundaries",
    ],
)
def test_package_error_message(
    package_name: str,
    actual_version: str | None,
    min_version: str | None,
    max_version: str | None,
    expected_message: str | None,
):
    message = _package_error_message(
        package_name=package_name,
        actual_package_version=version.parse(actual_version)
        if actual_version
        else None,
        min_package_version=version.parse(min_version) if min_version else None,
        max_package_version=version.parse(max_version) if max_version else None,
    )
    assert message == expected_message


@pytest.mark.parametrize(
    "package_prerequisites, expected_messages",
    [
        (
            [
                PackagePrerequisites("package1", None, None),
                PackagePrerequisites("package2", None, None),
                PackagePrerequisites("package3", None, None),
            ],
            [],
        ),
        (
            [
                PackagePrerequisites("package1", None, None),
                PackagePrerequisites("incompatible_package2", None, None),
                PackagePrerequisites("incompatible_package3", None, None),
            ],
            ["error_incompatible_package2", "error_incompatible_package3"],
        ),
        (
            [
                PackagePrerequisites("incompatible_package1", None, None),
                PackagePrerequisites("incompatible_package2", None, None),
                PackagePrerequisites("incompatible_package3", None, None),
            ],
            [
                "error_incompatible_package1",
                "error_incompatible_package2",
                "error_incompatible_package3",
            ],
        ),
    ],
    ids=[
        "no_error_messages",
        "some_error_messages",
        "all_error_messages",
    ],
)
def test_package_prerequisites_error_messages(
    package_prerequisites, expected_messages, monkeypatch
):
    def mock_package_error_message(package_name, **unused_args):
        return f"error_{package_name}" if "incompatible" in package_name else None

    monkeypatch.setattr(
        "oumi.utils.packaging._package_error_message", mock_package_error_message
    )

    error_messages = _package_prerequisites_error_messages(package_prerequisites)
    assert error_messages == expected_messages


def test_check_package_prerequisites():
    package_prerequisites = [
        PackagePrerequisites("pytest", "80.0.0", None),
        PackagePrerequisites("non_existing", "0.0.1"),
    ]
    pytest_version = importlib.metadata.version("pytest")
    expected_runtime_error_str = (
        "The current run cannot be launched because the platform prerequisites are not "
        "satisfied. In order to proceed, the following package(s) must be installed "
        "and have the correct version:\n"
        f"Package `pytest` version is {pytest_version}, which is not compatible. "
        "Please install: version >= 80.0.0.\n"
        "Package `non_existing` is not installed. Please install: version >= 0.0.1."
    )
    with pytest.raises(RuntimeError) as runtime_error:
        check_package_prerequisites(package_prerequisites)
    assert runtime_error.value.args[0] == expected_runtime_error_str


def test_verify_trl_vllm_compatibility_skips_when_vllm_missing(monkeypatch):
    def mock_version(pkg):
        if pkg == "vllm":
            raise importlib.metadata.PackageNotFoundError
        return "0.26.0"

    monkeypatch.setattr("importlib.metadata.version", mock_version)
    verify_trl_vllm_compatibility("test")  # Should not raise


def test_verify_trl_vllm_compatibility_passes_when_compatible(monkeypatch):
    def mock_version(pkg):
        return {"vllm": "0.14.0", "trl": "0.29.0"}[pkg]

    monkeypatch.setattr("importlib.metadata.version", mock_version)
    verify_trl_vllm_compatibility("test")  # Should not raise


def test_verify_trl_vllm_compatibility_fails_old_trl_new_vllm(monkeypatch):
    def mock_version(pkg):
        return {"vllm": "0.14.0", "trl": "0.26.0"}[pkg]

    monkeypatch.setattr("importlib.metadata.version", mock_version)
    with pytest.raises(RuntimeError, match="vLLM < 0.12.0"):
        verify_trl_vllm_compatibility("test")


def test_verify_trl_vllm_compatibility_fails_new_trl_old_vllm(monkeypatch):
    def mock_version(pkg):
        return {"vllm": "0.10.2", "trl": "0.29.0"}[pkg]

    monkeypatch.setattr("importlib.metadata.version", mock_version)
    with pytest.raises(RuntimeError, match="vLLM >= 0.11.0"):
        verify_trl_vllm_compatibility("test")


def _vllm_specifiers_by_extra() -> dict[str, str]:
    """Return the vLLM version specifier for each optional-dependencies extra.

    Parses the repo's pyproject.toml and walks `[project.optional-dependencies]`,
    returning for each extra that pins vLLM the normalized specifier string
    (`str(Requirement(dep).specifier)`) of the dep whose name is "vllm".

    Scope is `[project.optional-dependencies]` ONLY. A vllm pin added to the base
    `[project.dependencies]` list or a `[tool.uv]` override is intentionally out of
    this helper's scope.
    """
    # tomllib is stdlib only on py3.11+; import lazily so this module still imports
    # under py3.10 (the tests that call this are skipped there).
    import tomllib

    pyproject_path = Path(__file__).parents[3] / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    specifiers_by_extra: dict[str, str] = {}
    optional_deps = pyproject["project"]["optional-dependencies"]
    for extra, deps in optional_deps.items():
        for dep in deps:
            requirement = Requirement(dep)
            if requirement.name == "vllm":
                specifiers_by_extra[extra] = str(requirement.specifier)
                break
    return specifiers_by_extra


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib is py3.11+; oumi supports 3.10, CI runs 3.11",
)
def test_vllm_pin_gpu_matches_ci_cpu():
    """The `gpu` and `ci_cpu` extras resolve to an identical specifier set.

    Compared as normalized specifier strings (not byte-identity), so semantically
    equal pins written differently still pass while a real divergence fails.
    """
    specifiers = _vllm_specifiers_by_extra()
    assert specifiers["gpu"] == specifiers["ci_cpu"]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib is py3.11+; oumi supports 3.10, CI runs 3.11",
)
def test_vllm_floor_at_least_0_19():
    """Every pinned extra floors vLLM at >= 0.19 (0.19 first serves nemotron_h)."""
    for extra, specifier_str in _vllm_specifiers_by_extra().items():
        specifier = Requirement(f"vllm{specifier_str}").specifier
        lower_bounds = [
            Version(clause.version)
            for clause in specifier
            if clause.operator in (">=", ">")
        ]
        assert lower_bounds, f"extra `{extra}` has no vLLM lower bound"
        assert min(lower_bounds) >= Version("0.19")


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="tomllib is py3.11+; oumi supports 3.10, CI runs 3.11",
)
def test_exactly_two_explicit_vllm_pins():
    """Exactly two extras carry an explicit vLLM pin: `gpu` and `ci_cpu`.

    `ci_gpu` pulls vLLM transitively via `oumi[...gpu...]` with no explicit pin;
    a silent third explicit pin would fail this test.
    """
    assert set(_vllm_specifiers_by_extra()) == {"gpu", "ci_cpu"}
