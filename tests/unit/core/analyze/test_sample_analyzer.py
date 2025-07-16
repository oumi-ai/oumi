"""Unit tests for the sample analyzer plugin system."""

import pytest

from oumi.core.analyze.sample_analyzer import AnalyzerRegistry, SampleAnalyzer


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test to ensure test isolation."""
    AnalyzerRegistry._analyzers.clear()
    yield


class SimpleAnalyzer(SampleAnalyzer):
    """Simple analyzer for testing."""

    def analyze_message(self, text_content: str, message_metadata: dict) -> dict:
        return {"length": len(text_content)}


def test_analyzer_registry_register():
    """Test registering an analyzer."""
    AnalyzerRegistry.register("simple", SimpleAnalyzer)
    assert "simple" in AnalyzerRegistry._analyzers


def test_analyzer_registry_get_analyzer():
    """Test getting a registered analyzer."""
    AnalyzerRegistry.register("simple", SimpleAnalyzer)
    analyzer_class = AnalyzerRegistry.get_analyzer("simple")
    assert analyzer_class == SimpleAnalyzer


def test_analyzer_registry_create_analyzer():
    """Test creating an analyzer instance."""
    AnalyzerRegistry.register("simple", SimpleAnalyzer)
    analyzer = AnalyzerRegistry.create_analyzer("simple")
    assert isinstance(analyzer, SimpleAnalyzer)


def test_analyzer_analyze_message():
    """Test that analyzer can analyze a message."""
    AnalyzerRegistry.register("simple", SimpleAnalyzer)
    analyzer = AnalyzerRegistry.create_analyzer("simple")
    result = analyzer.analyze_message("hello", {"role": "user"})
    assert result == {"length": 5}


def test_analyzer_registry_unknown():
    """Test handling of unknown analyzer."""
    assert AnalyzerRegistry.get_analyzer("unknown") is None

    with pytest.raises(ValueError, match="Unknown analyzer ID: unknown"):
        AnalyzerRegistry.create_analyzer("unknown")


def test_analyzer_registry_duplicate_id():
    """Test that registering the same analyzer ID twice raises an error."""
    # Register analyzer for the first time
    AnalyzerRegistry.register("duplicate", SimpleAnalyzer)
    assert "duplicate" in AnalyzerRegistry._analyzers

    # Try to register the same ID again
    with pytest.raises(
        ValueError, match="Analyzer ID 'duplicate' is already registered"
    ):
        AnalyzerRegistry.register("duplicate", SimpleAnalyzer)

    # Verify the original registration is still intact
    analyzer = AnalyzerRegistry.create_analyzer("duplicate")
    assert isinstance(analyzer, SimpleAnalyzer)
