# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for the analyze module with user-defined tests.

These tests verify that the declarative test system works end-to-end,
from config loading through analysis to report generation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from oumi.core.analyze import DatasetAnalyzer, HTMLReportGenerator
from oumi.core.configs import AnalyzeConfig, SampleAnalyzerParams
from oumi.core.configs.params.test_params import TestParams
from tests import get_configs_dir


@pytest.fixture
def mock_dataset_file(tmp_path: Path) -> Path:
    """Create a mock JSONL dataset file for testing."""
    mock_data = []
    for i in range(30):
        mock_data.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Question number {i}: What is the meaning of life and happiness?",
                    },
                    {
                        "role": "assistant",
                        "content": f"Response {i}: The meaning of life is subjective and varies from person to person. Some find meaning in relationships.",
                    },
                ]
            }
        )
    # Add some edge cases
    mock_data.append(
        {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
            ]
        }
    )

    data_file = tmp_path / "mock_data.jsonl"
    with open(data_file, "w") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")

    return data_file


class TestAnalyzeWithTests:
    """Integration tests for analyze with user-defined tests."""

    @pytest.fixture
    def sample_config_with_tests(
        self, tmp_path: Path, mock_dataset_file: Path
    ) -> AnalyzeConfig:
        """Create a sample config with tests for integration testing.

        Uses mock data with minimal analyzers and several test types.
        Note: Column names follow the pattern {source}__{analyzer_id}__{metric}.
        """
        return AnalyzeConfig(
            dataset_path=str(mock_dataset_file),
            split="train",
            output_path=str(tmp_path / "test_output"),
            analyzers=[
                SampleAnalyzerParams(
                    id="diversity",
                    params={},
                ),
                SampleAnalyzerParams(
                    id="format",
                    params={},
                ),
            ],
            tests=[
                # Threshold test - check diversity ratio
                TestParams(
                    id="diversity_check",
                    type="threshold",
                    metric="text_content__diversity__unique_words_ratio",
                    operator=">",
                    value=0.1,
                    min_percentage=80.0,
                    severity="medium",
                    title="Vocabulary diversity check",
                    description="Messages should have >10% unique words",
                ),
                # Distribution test - role balance
                TestParams(
                    id="balanced_roles",
                    type="distribution",
                    metric="role",
                    check="max_fraction",
                    threshold=0.9,  # Allow up to 90% of one role
                    severity="low",
                    title="Role distribution check",
                ),
                # Regex test - special tokens
                TestParams(
                    id="no_special_tokens",
                    type="regex",
                    text_field="text_content",
                    pattern=r"<\|(?:endoftext|im_start|im_end)\|>",
                    max_percentage=0.0,
                    severity="high",
                    title="Special token leakage",
                ),
            ],
            generate_report=True,
            report_title="Integration Test Report",
        )

    @pytest.fixture
    def example_config_path(self) -> Path:
        """Get the path to the example config file."""
        return get_configs_dir() / "examples" / "analyze" / "analyze_with_tests.yaml"

    def test_example_config_exists(self, example_config_path: Path):
        """Verify the example config file exists."""
        assert example_config_path.exists(), (
            f"Example config not found: {example_config_path}"
        )

    def test_example_config_loads(self, example_config_path: Path):
        """Test that the example config file loads correctly."""
        config = AnalyzeConfig.from_yaml(example_config_path)

        assert config.dataset_name == "tatsu-lab/alpaca"
        assert len(config.analyzers) > 0
        assert len(config.tests) > 0, "Config should have tests defined"

        # Check test types are properly configured
        test_types = {t.type for t in config.tests}
        expected_types = {"threshold", "percentage", "distribution", "regex", "query"}
        assert expected_types.issubset(test_types), (
            f"Missing expected test types. Found: {test_types}, expected: {expected_types}"
        )

    def test_analyze_with_tests_runs(
        self, sample_config_with_tests: AnalyzeConfig, tmp_path: Path
    ):
        """Test that analysis with tests runs end-to-end."""
        config = sample_config_with_tests

        # Run analysis
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        # Verify analysis completed
        assert analyzer.analysis_summary is not None
        assert analyzer.message_df is not None
        assert not analyzer.message_df.empty

        # Verify test results are present in summary
        summary = analyzer.analysis_summary
        assert "test_summary" in summary, "test_summary should be in analysis summary"

        test_summary = summary["test_summary"]
        assert "total_tests" in test_summary
        assert "passed_tests" in test_summary
        assert "failed_tests" in test_summary
        assert "results" in test_summary

        # Verify we have the expected number of tests
        assert test_summary["total_tests"] == len(config.tests)

    def test_analyze_generates_report_with_tests(
        self, sample_config_with_tests: AnalyzeConfig, tmp_path: Path
    ):
        """Test that report generation includes test results."""
        config = sample_config_with_tests

        # Run analysis
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        # Generate report
        report_generator = HTMLReportGenerator()
        report_path = report_generator.generate_report(
            analyzer=analyzer,
            output_path=tmp_path / "report",
            title="Test Report with Tests",
        )

        # Verify report was created
        assert report_path.exists()
        index_html = report_path / "index.html"
        assert index_html.exists(), "index.html should be created"

        # Check report contains test results section
        html_content = index_html.read_text()
        assert "Test Results" in html_content, "Report should have Test Results section"

    def test_analyze_test_results_structure(
        self, sample_config_with_tests: AnalyzeConfig, tmp_path: Path
    ):
        """Test that test results have the correct structure."""
        config = sample_config_with_tests

        # Run analysis
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        summary = analyzer.analysis_summary
        test_summary = summary["test_summary"]

        # Check results structure
        results = test_summary["results"]
        assert isinstance(results, list)
        assert len(results) > 0

        # Check each result has required fields
        required_fields = [
            "test_id",
            "test_type",
            "passed",
            "severity",
            "title",
            "description",
        ]

        for result in results:
            for field in required_fields:
                assert field in result, f"Result missing required field: {field}"

            # Verify field types
            assert isinstance(result["passed"], bool)
            assert result["severity"] in ("high", "medium", "low")
            assert result["test_type"] in (
                "threshold",
                "percentage",
                "distribution",
                "regex",
                "contains",
                "contains-any",
                "contains-all",
                "query",
                "outliers",
                "composite",
                "python",
            )

    def test_analyze_passed_and_failed_lists(
        self, sample_config_with_tests: AnalyzeConfig, tmp_path: Path
    ):
        """Test that passed and failed test lists are populated correctly."""
        config = sample_config_with_tests

        # Run analysis
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        summary = analyzer.analysis_summary

        # Check tests_passed and tests_failed are present
        assert "tests_passed" in summary
        assert "tests_failed" in summary

        tests_passed = summary["tests_passed"]
        tests_failed = summary["tests_failed"]

        assert isinstance(tests_passed, list)
        assert isinstance(tests_failed, list)

        # Total should equal the number of tests
        test_summary = summary["test_summary"]
        assert len(tests_passed) + len(tests_failed) == test_summary["total_tests"]

    def test_threshold_test_execution(self, tmp_path: Path, mock_dataset_file: Path):
        """Test that threshold tests execute correctly."""
        config = AnalyzeConfig(
            dataset_path=str(mock_dataset_file),
            split="train",
            output_path=str(tmp_path / "threshold_test"),
            analyzers=[
                SampleAnalyzerParams(id="diversity", params={}),
            ],
            tests=[
                TestParams(
                    id="diversity_threshold",
                    type="threshold",
                    metric="text_content__diversity__unique_words_ratio",
                    operator=">",
                    value=0,  # All messages should have > 0 unique ratio
                    min_percentage=90.0,
                    severity="high",
                    title="Diversity threshold check",
                ),
            ],
        )

        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        summary = analyzer.analysis_summary
        test_summary = summary["test_summary"]

        assert test_summary["total_tests"] == 1
        results = test_summary["results"]
        assert len(results) == 1

    def test_query_test_execution(self, tmp_path: Path, mock_dataset_file: Path):
        """Test that query tests execute correctly with pandas expressions."""
        config = AnalyzeConfig(
            dataset_path=str(mock_dataset_file),
            split="train",
            output_path=str(tmp_path / "query_test"),
            analyzers=[
                SampleAnalyzerParams(id="diversity", params={}),
            ],
            tests=[
                TestParams(
                    id="diversity_query",
                    type="query",
                    expression="text_content__diversity__unique_words_ratio > 0.1",
                    min_percentage=80.0,
                    severity="medium",
                    title="Messages with good diversity",
                ),
            ],
        )

        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        summary = analyzer.analysis_summary
        test_summary = summary["test_summary"]

        assert test_summary["total_tests"] == 1
        results = test_summary["results"]
        assert len(results) == 1
        assert results[0]["test_type"] == "query"

    def test_distribution_test_execution(self, tmp_path: Path, mock_dataset_file: Path):
        """Test that distribution tests execute correctly."""
        config = AnalyzeConfig(
            dataset_path=str(mock_dataset_file),
            split="train",
            output_path=str(tmp_path / "dist_test"),
            analyzers=[
                SampleAnalyzerParams(id="format", params={}),  # Minimal analyzer
            ],
            tests=[
                TestParams(
                    id="role_distribution",
                    type="distribution",
                    metric="role",
                    check="max_fraction",
                    threshold=0.99,  # Very permissive
                    severity="low",
                    title="Role distribution",
                ),
            ],
        )

        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        summary = analyzer.analysis_summary
        test_summary = summary["test_summary"]

        assert test_summary["total_tests"] == 1
        results = test_summary["results"]
        assert len(results) == 1
        assert results[0]["test_type"] == "distribution"

    def test_composite_test_execution(self, tmp_path: Path, mock_dataset_file: Path):
        """Test that composite tests execute correctly."""
        config = AnalyzeConfig(
            dataset_path=str(mock_dataset_file),
            split="train",
            output_path=str(tmp_path / "composite_test"),
            analyzers=[
                SampleAnalyzerParams(id="diversity", params={}),
            ],
            tests=[
                TestParams(
                    id="composite_quality",
                    type="composite",
                    composite_operator="any",
                    severity="medium",
                    title="Composite quality check",
                    tests=[
                        {
                            "type": "threshold",
                            "metric": "text_content__diversity__unique_words_ratio",
                            "operator": ">",
                            "value": 0.1,
                            "min_percentage": 80.0,
                        },
                        {
                            "type": "distribution",
                            "metric": "role",
                            "check": "max_fraction",
                            "threshold": 0.9,
                        },
                    ],
                ),
            ],
        )

        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        summary = analyzer.analysis_summary
        test_summary = summary["test_summary"]

        assert test_summary["total_tests"] == 1
        results = test_summary["results"]
        assert len(results) == 1
        assert results[0]["test_type"] == "composite"

    def test_report_data_files_include_test_summary(
        self, sample_config_with_tests: AnalyzeConfig, tmp_path: Path
    ):
        """Test that generated report data files contain test information."""
        config = sample_config_with_tests

        # Run analysis
        analyzer = DatasetAnalyzer(config)
        analyzer.analyze_dataset()

        # Generate report
        report_generator = HTMLReportGenerator()
        report_path = report_generator.generate_report(
            analyzer=analyzer,
            output_path=tmp_path / "report_data_test",
        )

        # Check data directory exists
        data_dir = report_path / "data"
        assert data_dir.exists(), "Data directory should be created"

        # Read the HTML and check for test_results_summary
        index_html = report_path / "index.html"
        html_content = index_html.read_text()

        # The test results should appear in the HTML
        assert "pass_rate" in html_content.lower() or "Pass Rate" in html_content


class TestAnalyzeConfigWithTests:
    """Tests for AnalyzeConfig with tests field."""

    def test_config_with_tests_serialization(self, tmp_path: Path):
        """Test that config with tests can be serialized to/from YAML."""
        config = AnalyzeConfig(
            dataset_name="test_dataset",
            split="train",
            sample_count=10,
            output_path=str(tmp_path),
            analyzers=[
                SampleAnalyzerParams(id="length", params={"char_count": True}),
            ],
            tests=[
                TestParams(
                    id="test1",
                    type="threshold",
                    metric="length__char_count",
                    operator=">",
                    value=0,
                    max_percentage=5.0,
                    severity="high",
                    title="Test title",
                ),
            ],
        )

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(yaml_path)

        # Load back
        loaded = AnalyzeConfig.from_yaml(yaml_path)

        assert loaded.dataset_name == config.dataset_name
        assert len(loaded.tests) == 1
        assert loaded.tests[0].id == "test1"
        assert loaded.tests[0].type == "threshold"
        assert loaded.tests[0].severity == "high"

    def test_config_validates_test_ids_unique(self, tmp_path: Path):
        """Test that config validation catches duplicate test IDs."""
        with pytest.raises(ValueError, match="Duplicate test ID"):
            AnalyzeConfig(
                dataset_name="test_dataset",
                split="train",
                sample_count=10,
                output_path=str(tmp_path),
                tests=[
                    TestParams(
                        id="duplicate_id",
                        type="threshold",
                        metric="some_metric",
                        operator=">",
                        value=0,
                    ),
                    TestParams(
                        id="duplicate_id",  # Same ID - should fail
                        type="threshold",
                        metric="another_metric",
                        operator=">",
                        value=0,
                    ),
                ],
            )
