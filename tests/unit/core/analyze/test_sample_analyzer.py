"""Unit tests for the sample analyzer plugin system."""

import pytest

from oumi.core.analyze.sample_analyzer import AnalyzerRegistry, SampleAnalyzer


class SimpleAnalyzer(SampleAnalyzer):
    """Simple analyzer for testing."""

    def analyze_message(self, text_content: str, message_metadata: dict) -> dict:
        return {"length": len(text_content)}


def test_analyzer_registry_basic():
    """Test basic registry functionality."""
    # Clear registry
    AnalyzerRegistry._analyzers.clear()

    # Register analyzer
    AnalyzerRegistry.register("simple", SimpleAnalyzer)
    assert "simple" in AnalyzerRegistry._analyzers

    # Get analyzer
    analyzer_class = AnalyzerRegistry.get_analyzer("simple")
    assert analyzer_class == SimpleAnalyzer

    # Create instance
    analyzer = AnalyzerRegistry.create_analyzer("simple", {"test": "config"})
    assert isinstance(analyzer, SimpleAnalyzer)
    assert analyzer.config == {"test": "config"}

    # Test analysis
    result = analyzer.analyze_message("hello", {"role": "user"})
    assert result == {"length": 5}


def test_analyzer_registry_unknown():
    """Test handling of unknown analyzer."""
    AnalyzerRegistry._analyzers.clear()

    assert AnalyzerRegistry.get_analyzer("unknown") is None

    with pytest.raises(ValueError, match="Unknown analyzer ID: unknown"):
        AnalyzerRegistry.create_analyzer("unknown", {})
