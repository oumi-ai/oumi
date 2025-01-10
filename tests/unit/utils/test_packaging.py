import importlib.metadata
from typing import Union

import pytest

from oumi.utils.packaging import (
    PackagePrerequisites,
    _package_error_message,
    _package_prerequisites_error_messages,
    check_package_prerequisites,
)


@pytest.mark.parametrize(
    "package_name, actual_version, comparator, required_version, expected_message",
    [
        # Happy path tests (package installed, correct version).
        ("my_package", "1.23.5", ">=", "1.20.8", None),
        ("my_package", "1.43.11", ">", "1.43.1", None),
        ("my_package", "2.0.1", "==", "2.0.1", None),
        ("my_package", "2.28.1", "==", None, None),
        ("my_package", "2.28.1", None, "2.28.1", None),
        ("my_package", "2.28.1", None, None, None),
        # Error cases (package not installed).
        ("my_package", None, None, None, "Package `my_package` is not installed."),
        (
            "my_package",
            None,
            ">=",
            "1.20.0",
            "Package `my_package` is not installed. Please install version >=1.20.0.",
        ),
        # Error cases (package version incompatible).
        (
            "my_package",
            "1.20.0",
            ">=",
            "1.23.5",
            "Package `my_package` version is 1.20.0 but >=1.23.5 is required. "
            "Please install version >=1.23.5.",
        ),
        (
            "my_package",
            "1.4.3",
            "==",
            "2.0.1",
            "Package `my_package` version is 1.4.3 but ==2.0.1 is required. "
            "Please install version ==2.0.1.",
        ),
    ],
    ids=[
        "happy_path_greater_equal",
        "happy_path_greater",
        "happy_path_equal",
        "happy_path_no_required_version",
        "happy_path_no_comparator",
        "happy_path_no_required_version_no_comparator",
        "error_case_package_not_installed",
        "error_case_package_not_installed_specific_version_required",
        "error_case_package_version_incompatible_too_low",
        "error_case_package_version_incompatible_not_equal",
    ],
)
def test_package_error_message(
    package_name: str,
    actual_version: Union[str, None],
    comparator: Union[str, None],
    required_version: Union[str, None],
    expected_message: Union[str, None],
):
    message = _package_error_message(
        package_name=package_name,
        actual_version=actual_version,
        comparator=comparator,
        required_version=required_version,
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
        PackagePrerequisites("pytest", "100", ">="),
        PackagePrerequisites("non_existing", "0.0.1", ">"),
    ]
    pytest_version = importlib.metadata.version("pytest")
    expected_runtime_error_str = (
        "The current run cannot be launched because the platform prerequisites are not "
        "satisfied. In order to proceed, the following package(s) must be installed "
        "and have the correct version:\n"
        f"Package `pytest` version is {pytest_version} but >=100 is required. "
        "Please install version >=100.\n"
        "Package `non_existing` is not installed. Please install version >0.0.1."
    )
    with pytest.raises(RuntimeError) as runtime_error:
        check_package_prerequisites(package_prerequisites)
    assert runtime_error.value.args[0] == expected_runtime_error_str
