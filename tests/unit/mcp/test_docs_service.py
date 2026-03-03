# pyright: reportReturnType=false
"""Tests for oumi.mcp.docs_service — docstring parsing, scoring, search."""

import pytest

from oumi.mcp.docs_service import (
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
