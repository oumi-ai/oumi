"""Sample analyzer plugin system for OUMI.

This package provides a plugin-based architecture for analyzing conversation data
with different types of sample analyzers (length, safety, etc.).
"""

# Import base classes
from oumi.core.analyze.sample_analyzer import AnalyzerRegistry, SampleAnalyzer

__all__ = ["AnalyzerRegistry", "SampleAnalyzer"]
