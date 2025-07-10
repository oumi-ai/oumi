"""Analyzer plugin system for OUMI.

This package provides a plugin-based architecture for analyzing conversation data
with different types of analyzers (length, safety, etc.).
"""

# Import base classes
from oumi.core.analyzers.base import AnalyzerRegistry

# Import and register analyzer implementations
try:
    from oumi.core.analyzers.length_analyzer import LengthAnalyzer

    AnalyzerRegistry.register("length", LengthAnalyzer)
except ImportError:
    pass

__all__ = ["AnalyzerRegistry"]
