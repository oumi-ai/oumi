#!/usr/bin/env python3
"""
Simple test to verify Allure setup works correctly.
Run this locally to test Allure integration.
"""

import os
import sys

import allure
import pytest


@allure.epic("Oumi E2E Tests")
@allure.feature("Allure Integration")
class TestAllureSetup:
    @allure.story("Basic Allure Test")
    @allure.severity(allure.severity_level.NORMAL)
    def test_allure_basic(self):
        """Test that Allure basic functionality works."""
        with allure.step("Step 1: Basic assertion"):
            assert 1 + 1 == 2

        with allure.step("Step 2: String operation"):
            result = "hello" + " world"
            assert result == "hello world"

        with allure.step("Step 3: List operation"):
            items = [1, 2, 3]
            items.append(4)
            assert len(items) == 4

    @allure.story("Allure Attachments")
    @allure.severity(allure.severity_level.NORMAL)
    def test_allure_attachments(self):
        """Test that Allure can handle attachments."""
        with allure.step("Create test data"):
            test_data = "This is test data for Allure attachment"

        with allure.step("Attach text data"):
            allure.attach(
                test_data,
                name="test_data.txt",
                attachment_type=allure.attachment_type.TEXT,
            )

        with allure.step("Create JSON data"):
            json_data = '{"test": "data", "number": 42}'

        with allure.step("Attach JSON data"):
            allure.attach(
                json_data,
                name="test_data.json",
                attachment_type=allure.attachment_type.JSON,
            )

        assert True

    @allure.story("Allure Links")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.link("https://github.com/oumi-ai/oumi", name="Oumi Repository")
    @allure.issue("123", "Test Issue")
    @allure.testcase("TC-001", "Test Case 001")
    def test_allure_links(self):
        """Test that Allure can handle links and issue tracking."""
        with allure.step("Test with links"):
            assert True

    @allure.story("Allure Descriptions")
    @allure.severity(allure.severity_level.NORMAL)
    @allure.description("""
    This is a detailed test description.
    It can contain multiple lines and formatting.

    Features:
    - Multi-line descriptions
    - Markdown formatting
    - Rich text support
    """)
    def test_allure_descriptions(self):
        """Test that Allure can handle detailed descriptions."""
        with allure.step("Test description functionality"):
            assert True

    @allure.story("Allure Parameters")
    @allure.severity(allure.severity_level.NORMAL)
    @pytest.mark.parametrize(
        "input_value,expected",
        [
            (1, 2),
            (2, 4),
            (3, 6),
        ],
    )
    def test_allure_parameters(self, input_value, expected):
        """Test that Allure can handle parameterized tests."""
        with allure.step(f"Test with input: {input_value}"):
            result = input_value * 2
            assert result == expected

    @allure.story("Allure Environment")
    @allure.severity(allure.severity_level.NORMAL)
    def test_allure_environment(self):
        """Test that Allure can capture environment information."""
        with allure.step("Capture environment info"):
            allure.dynamic.description(f"""
            Environment Information:
            - Python Version: {sys.version}
            - Platform: {sys.platform}
            - Working Directory: {os.getcwd()}
            """)

        with allure.step("Test environment capture"):
            assert sys.platform is not None
            assert os.getcwd() is not None


@allure.epic("Oumi E2E Tests")
@allure.feature("Error Handling")
class TestAllureErrors:
    @allure.story("Failed Test")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_allure_failure(self):
        """Test that Allure can handle test failures."""
        with allure.step("Step that will fail"):
            # This will fail
            assert 1 == 2, "This test is designed to fail"

    @allure.story("Exception Test")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_allure_exception(self):
        """Test that Allure can handle exceptions."""
        with allure.step("Step that will raise exception"):
            # This will raise an exception
            raise ValueError("This test is designed to raise an exception")


if __name__ == "__main__":
    # Run the tests with Allure
    import subprocess
    import sys

    print("Running Allure tests...")

    # Create allure-results directory
    os.makedirs("allure-results", exist_ok=True)

    # Run pytest with Allure
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        __file__,
        "--alluredir=allure-results",
        "--allure-link-pattern=issue:https://github.com/oumi-ai/oumi/issues/{}",
        "--allure-link-pattern=tms:https://github.com/oumi-ai/oumi/issues/{}",
        "-v",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print(f"Exit code: {result.returncode}")

    if result.returncode == 0:
        print("\n✅ Allure tests completed successfully!")
        print("📊 To view the report:")
        print("   allure serve allure-results")
    else:
        print("\n❌ Some tests failed (this is expected for error tests)")
        print("📊 To view the report:")
        print("   allure serve allure-results")
