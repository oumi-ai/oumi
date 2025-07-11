"""Sample analyzer plugin system for OUMI.

This package provides a plugin-based architecture for analyzing conversation data
with different types of sample analyzers (length, safety, etc.).
"""

# Import base classes
from oumi.core.analyze.text_analyzer import AnalyzerRegistry, SampleAnalyzer

# Import and register sample analyzer implementations
try:
    from oumi.core.analyze.length_analyzer import LengthAnalyzer

    AnalyzerRegistry.register("length", LengthAnalyzer)
except ImportError:
    pass

__all__ = ["AnalyzerRegistry", "SampleAnalyzer"]
