# pyright: reportReturnType=false, reportPrivateUsage=false
"""Tests for oumi.mcp.docs_service — docstring parsing, scoring, search, indexing."""

import dataclasses

import pytest

from oumi.mcp.docs_service import (
    _build_field_docs,
    _extract_field_docstrings,
    _index_class,
    _index_module,
    _score_entry,
    parse_docstring,
    search_docs,
)


def test_parse_docstring_none():
    summary, sections = parse_docstring(None)
    assert summary == ""
    assert sections == []


def test_parse_docstring_empty():
    summary, sections = parse_docstring("")
    assert summary == ""
    assert sections == []


def test_parse_docstring_summary_only():
    summary, sections = parse_docstring("A simple summary.")
    assert summary == "A simple summary."
    assert sections == []


def test_parse_docstring_with_args_section():
    doc = "Summary line.\n\nArgs:\n    x: The x value.\n    y: The y value."
    summary, sections = parse_docstring(doc)
    assert summary == "Summary line."
    assert len(sections) == 1
    assert sections[0]["name"] == "Args"
    assert "x: The x value" in sections[0]["content"]


def test_parse_docstring_multiple_sections():
    doc = "Summary.\n\nArgs:\n    x: val\n\nReturns:\n    The result."
    summary, sections = parse_docstring(doc)
    assert summary == "Summary."
    assert len(sections) == 2
    assert sections[0]["name"] == "Args"
    assert sections[1]["name"] == "Returns"


def _blob(name="Foo", qual="mod.Foo", kind="class", summary="", fields=None):
    return {
        "name_lower": name.lower(),
        "qual_lower": qual.lower(),
        "kind_lower": kind.lower(),
        "summary_lower": summary.lower(),
        "field_names_lower": [f.lower() for f in (fields or [])],
        "section_contents_lower": [],
        "module_lower": "mod",
        "is_class_like": kind in ("class", "dataclass"),
    }


@pytest.mark.parametrize(
    "name,qual,kind,summary,fields,query,min_score",
    [
        (
            "TrainingConfig",
            "mod.TrainingConfig",
            "class",
            "",
            [],
            "trainingconfig",
            100,
        ),
        ("TrainingConfig", "mod.TrainingConfig", "class", "", [], "config", 50),
        ("Foo", "oumi.core.configs.TrainingConfig", "class", "", [], "oumi.core", 30),
        ("Foo", "mod.Foo", "class", "", ["learning_rate"], "learning_rate", 30),
        ("Foo", "mod.Foo", "class", "Configure training parameters", [], "training", 5),
    ],
)
def test_score_entry_threshold(name, qual, kind, summary, fields, query, min_score):
    blob = _blob(name=name, qual=qual, kind=kind, summary=summary, fields=fields)
    assert _score_entry(blob, query) >= min_score


def test_score_entry_no_match():
    assert _score_entry(_blob(name="Foo"), "zzz_nonexistent") == 0


def test_score_entry_class_boost():
    blob_class = _blob(name="Config", kind="class")
    blob_func = _blob(name="Config", kind="function")
    assert _score_entry(blob_class, "config") > _score_entry(blob_func, "config")


def test_search_docs_empty_query_error():
    r = search_docs([])
    assert r["error"] != ""


def test_search_docs_limit_validation():
    r = search_docs(["x"], limit=0)
    assert "limit" in r["error"]


@dataclasses.dataclass
class _SampleDataclass:
    """A sample dataclass for testing field docstring extraction."""

    name: str = "default"
    """The name field."""

    count: int = 0
    """How many items."""

    unlabeled: float = 1.0


def test_extract_field_docstrings_basic():
    result = _extract_field_docstrings(_SampleDataclass)
    assert result["name"] == "The name field."
    assert result["count"] == "How many items."
    assert "unlabeled" not in result


def test_extract_field_docstrings_non_dataclass():
    class Plain:
        x: int = 1

    result = _extract_field_docstrings(Plain)
    assert result == {}


def test_build_field_docs_dataclass():
    fields = _build_field_docs(_SampleDataclass)
    names = [f["name"] for f in fields]
    assert "name" in names
    assert "count" in names
    assert "unlabeled" in names

    name_field = next(f for f in fields if f["name"] == "name")
    assert name_field["description"] == "The name field."
    assert name_field["default"] == "'default'"

    count_field = next(f for f in fields if f["name"] == "count")
    assert count_field["description"] == "How many items."


def test_build_field_docs_non_dataclass():
    class NotADataclass:
        pass

    assert _build_field_docs(NotADataclass) == []


class _SampleClass:
    """A sample class with methods."""

    def public_method(self) -> str:
        """Return a greeting."""
        return "hello"

    def another_method(self, x: int) -> int:
        """Double the input."""
        return x * 2

    def _private_method(self) -> None:
        pass


def test_index_class_basic():
    entries = _index_class(_SampleClass, "test.module")

    assert len(entries) >= 3
    class_entry = entries[0]
    assert class_entry["name"] == "_SampleClass"
    assert class_entry["kind"] == "class"
    assert class_entry["module"] == "test.module"
    assert class_entry["qualified_name"] == "test.module._SampleClass"
    assert "A sample class" in class_entry["summary"]

    method_names = [e["name"] for e in entries if e["kind"] == "method"]
    assert "public_method" in method_names
    assert "another_method" in method_names
    assert "_private_method" not in method_names


def test_index_class_methods_have_parent():
    entries = _index_class(_SampleClass, "test.module")
    methods = [e for e in entries if e["kind"] == "method"]
    for m in methods:
        assert m["parent_class"] == "_SampleClass"


def test_index_class_dataclass():
    entries = _index_class(_SampleDataclass, "test.module")
    class_entry = entries[0]
    assert class_entry["kind"] == "dataclass"
    assert len(class_entry["fields"]) == 3


def test_index_module_real():
    """Index a small real oumi module and verify structure."""
    entries, info = _index_module("oumi.mcp.models", "MCP data models")
    assert len(entries) > 0
    assert info["module"] == "oumi.mcp.models"
    assert info["description"] == "MCP data models"
    assert info["class_count"] > 0

    for entry in entries:
        assert entry["module"] == "oumi.mcp.models"


def test_index_module_nonexistent():
    """Indexing a nonexistent module returns empty results."""
    entries, info = _index_module("oumi.nonexistent.module", "Does not exist")
    assert entries == []
    assert info["class_count"] == 0
    assert info["function_count"] == 0
