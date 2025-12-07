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

"""Tests for the HTMLReportGenerator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from oumi.core.analyze.report_generator import HTMLReportGenerator


class TestHTMLReportGeneratorInit:
    """Tests for HTMLReportGenerator initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        generator = HTMLReportGenerator()
        assert generator.include_tables is True
        assert generator.include_recommendations is True
        assert generator.chart_height == 400
        assert generator.max_charts == 10

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        generator = HTMLReportGenerator(
            include_charts=False,
            include_tables=False,
            include_recommendations=False,
            chart_height=600,
            max_charts=5,
        )
        assert generator.include_charts is False
        assert generator.include_tables is False
        assert generator.include_recommendations is False
        assert generator.chart_height == 600
        assert generator.max_charts == 5

    def test_template_loaded(self):
        """Test that template is loaded correctly."""
        generator = HTMLReportGenerator()
        assert generator._template is not None


class TestHTMLReportGeneratorGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock analyzer with analysis results."""
        analyzer = MagicMock()
        analyzer.analysis_summary = {
            "dataset_overview": {
                "dataset_name": "test_dataset",
                "total_conversations": 100,
                "conversations_analyzed": 100,
                "dataset_coverage_percentage": 100.0,
                "total_messages": 200,
                "analyzers_used": ["length"],
            },
            "message_level_summary": {
                "length": {
                    "text_content_char_count": {
                        "count": 200,
                        "mean": 100.0,
                        "std": 20.0,
                        "min": 10.0,
                        "max": 500.0,
                        "median": 95.0,
                    }
                }
            },
            "conversation_level_summary": {},
            "conversation_turns": {
                "count": 100,
                "mean": 2.0,
                "std": 0.5,
                "min": 1,
                "max": 5,
                "median": 2.0,
            },
            "recommendations": [
                {
                    "category": "warning",
                    "severity": "medium",
                    "title": "Test Warning",
                    "description": "This is a test warning.",
                    "affected_samples": 10,
                    "metric_name": "char_count",
                }
            ],
        }

        # Create mock message_df
        analyzer.message_df = pd.DataFrame({
            "text_content": ["hello"] * 100,
            "role": ["user"] * 50 + ["assistant"] * 50,
            "text_content_length_char_count": [100] * 100,
        })

        return analyzer

    def test_generate_report_to_file(self, mock_analyzer):
        """Test generating report to a specific file."""
        generator = HTMLReportGenerator(include_charts=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            result = generator.generate_report(mock_analyzer, output_path)

            assert result == output_path
            assert output_path.exists()

            content = output_path.read_text()
            assert "test_dataset" in content
            assert "Test Warning" in content

    def test_generate_report_to_directory(self, mock_analyzer):
        """Test generating report to a directory."""
        generator = HTMLReportGenerator(include_charts=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            result = generator.generate_report(mock_analyzer, output_path)

            expected_file = output_path / "analysis_report.html"
            assert result == expected_file
            assert expected_file.exists()

    def test_generate_report_custom_title(self, mock_analyzer):
        """Test generating report with custom title."""
        generator = HTMLReportGenerator(include_charts=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generator.generate_report(
                mock_analyzer, output_path, title="Custom Report Title"
            )

            content = output_path.read_text()
            assert "Custom Report Title" in content

    def test_generate_report_without_recommendations(self, mock_analyzer):
        """Test generating report without recommendations section."""
        generator = HTMLReportGenerator(
            include_charts=False, include_recommendations=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generator.generate_report(mock_analyzer, output_path)

            content = output_path.read_text()
            # The recommendation content should not be present
            assert "Test Warning" not in content

    def test_generate_report_analysis_not_run(self):
        """Test error when analysis has not been run."""
        generator = HTMLReportGenerator(include_charts=False)

        mock_analyzer = MagicMock()
        mock_analyzer.analysis_summary = property(
            fget=lambda self: (_ for _ in ()).throw(
                RuntimeError("Analysis has not been run yet.")
            )
        )
        type(mock_analyzer).analysis_summary = property(
            fget=lambda self: (_ for _ in ()).throw(
                RuntimeError("Analysis has not been run yet.")
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            with pytest.raises(RuntimeError, match="Analysis has not been run"):
                generator.generate_report(mock_analyzer, output_path)

    def test_generate_report_creates_parent_directories(self, mock_analyzer):
        """Test that parent directories are created if they don't exist."""
        generator = HTMLReportGenerator(include_charts=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path" / "report.html"
            result = generator.generate_report(mock_analyzer, output_path)

            assert result == output_path
            assert output_path.exists()


class TestHTMLReportGeneratorWithPlotly:
    """Tests for report generation with Plotly charts."""

    @pytest.fixture
    def mock_analyzer_with_data(self):
        """Create a mock analyzer with actual DataFrame data."""
        analyzer = MagicMock()
        analyzer.analysis_summary = {
            "dataset_overview": {
                "dataset_name": "test_dataset",
                "total_conversations": 100,
                "conversations_analyzed": 100,
                "dataset_coverage_percentage": 100.0,
                "total_messages": 200,
                "analyzers_used": ["length"],
            },
            "message_level_summary": {},
            "conversation_level_summary": {},
            "conversation_turns": {},
            "recommendations": [],
        }

        # Create actual DataFrame for chart generation
        analyzer.message_df = pd.DataFrame({
            "conversation_index": range(100),
            "conversation_id": [f"conv_{i}" for i in range(100)],
            "message_index": range(100),
            "message_id": [f"msg_{i}" for i in range(100)],
            "role": ["user"] * 50 + ["assistant"] * 50,
            "text_content": ["hello world"] * 100,
            "text_content_length_char_count": list(range(50, 150)),
            "text_content_length_word_count": list(range(10, 110)),
        })

        return analyzer

    @pytest.mark.skipif(
        not HTMLReportGenerator()._check_plotly(),
        reason="Plotly not installed"
    )
    def test_generate_report_with_charts(self, mock_analyzer_with_data):
        """Test generating report with Plotly charts."""
        generator = HTMLReportGenerator(include_charts=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generator.generate_report(mock_analyzer_with_data, output_path)

            content = output_path.read_text()
            # Check that Plotly script is included
            assert "Plotly.newPlot" in content
            # Check that chart containers are present
            assert "chart_" in content

    @pytest.mark.skipif(
        not HTMLReportGenerator()._check_plotly(),
        reason="Plotly not installed"
    )
    def test_chart_generation_respects_max_charts(self, mock_analyzer_with_data):
        """Test that max_charts limit is respected."""
        generator = HTMLReportGenerator(include_charts=True, max_charts=2)

        charts = generator._generate_charts(mock_analyzer_with_data)

        # Should have at most 2 charts
        assert len(charts) <= 2


class TestHTMLReportGeneratorEdgeCases:
    """Tests for edge cases."""

    def test_empty_summary(self):
        """Test handling of empty analysis summary."""
        generator = HTMLReportGenerator(include_charts=False)

        mock_analyzer = MagicMock()
        mock_analyzer.analysis_summary = {
            "dataset_overview": {},
            "message_level_summary": {},
            "conversation_level_summary": {},
            "conversation_turns": {},
            "recommendations": [],
        }
        mock_analyzer.message_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            # Should not crash
            generator.generate_report(mock_analyzer, output_path)
            assert output_path.exists()

    def test_no_recommendations(self):
        """Test report generation when there are no recommendations."""
        generator = HTMLReportGenerator(include_charts=False)

        mock_analyzer = MagicMock()
        mock_analyzer.analysis_summary = {
            "dataset_overview": {
                "dataset_name": "clean_dataset",
                "total_conversations": 100,
            },
            "message_level_summary": {},
            "conversation_level_summary": {},
            "conversation_turns": {},
            "recommendations": [],
        }
        mock_analyzer.message_df = pd.DataFrame()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"
            generator.generate_report(mock_analyzer, output_path)

            content = output_path.read_text()
            # Should show "no issues detected" message
            assert "looks good" in content.lower() or "no issues" in content.lower()
