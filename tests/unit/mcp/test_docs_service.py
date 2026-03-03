"""Tests for oumi.mcp.docs_service — docstring parsing, scoring, search."""

from oumi.mcp.docs_service import (
    _score_entry,
    get_module_list,
    parse_docstring,
    search_docs,
)


class TestParseDocstring:
    def test_none(self):
        summary, sections = parse_docstring(None)
        assert summary == ""
        assert sections == []

    def test_empty(self):
        summary, sections = parse_docstring("")
        assert summary == ""
        assert sections == []

    def test_summary_only(self):
        summary, sections = parse_docstring("A simple summary.")
        assert summary == "A simple summary."
        assert sections == []

    def test_with_args_section(self):
        doc = "Summary line.\n\nArgs:\n    x: The x value.\n    y: The y value."
        summary, sections = parse_docstring(doc)
        assert summary == "Summary line."
        assert len(sections) == 1
        assert sections[0]["name"] == "Args"
        assert "x: The x value" in sections[0]["content"]

    def test_multiple_sections(self):
        doc = "Summary.\n\nArgs:\n    x: val\n\nReturns:\n    The result."
        summary, sections = parse_docstring(doc)
        assert summary == "Summary."
        assert len(sections) == 2
        assert sections[0]["name"] == "Args"
        assert sections[1]["name"] == "Returns"


class TestScoreEntry:
    def _blob(self, name="Foo", qual="mod.Foo", kind="class", summary="", fields=None):
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

    def test_exact_name_start(self):
        blob = self._blob(name="TrainingConfig")
        assert _score_entry(blob, "trainingconfig") >= 100

    def test_partial_name(self):
        blob = self._blob(name="TrainingConfig")
        assert _score_entry(blob, "config") >= 50

    def test_qualified_name(self):
        blob = self._blob(qual="oumi.core.configs.TrainingConfig")
        assert _score_entry(blob, "oumi.core") >= 30

    def test_field_match(self):
        blob = self._blob(fields=["learning_rate", "max_steps"])
        assert _score_entry(blob, "learning_rate") >= 30

    def test_summary_match(self):
        blob = self._blob(summary="Configure training parameters")
        assert _score_entry(blob, "training") >= 5

    def test_no_match(self):
        blob = self._blob(name="Foo")
        assert _score_entry(blob, "zzz_nonexistent") == 0

    def test_class_boost(self):
        blob_class = self._blob(name="Config", kind="class")
        blob_func = self._blob(name="Config", kind="function")
        assert _score_entry(blob_class, "config") > _score_entry(blob_func, "config")


class TestSearchDocs:
    def test_empty_query_error(self):
        r = search_docs([])
        assert r["error"] != ""

    def test_limit_validation(self):
        r = search_docs(["x"], limit=0)
        assert "limit" in r["error"]

    def test_index_empty_before_build(self):
        # Reset index state for isolation
        import oumi.mcp.docs_service as ds

        old_ready = ds._index_ready.is_set()
        ds._index_ready.clear()
        with ds._index_lock:
            old_index = ds._index
            ds._index = []
        try:
            r = search_docs(["something"])
            assert "empty" in r["error"].lower() or "building" in r["error"].lower()
        finally:
            with ds._index_lock:
                ds._index = old_index
            if old_ready:
                ds._index_ready.set()


class TestGetModuleList:
    def test_returns_response_shape(self):
        r = get_module_list()
        assert "modules" in r
        assert "total_entries" in r
        assert "index_ready" in r
        assert "oumi_version" in r
