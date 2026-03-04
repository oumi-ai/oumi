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

"""Documentation indexing and search service for Oumi's Python API.

Dynamically imports and introspects Oumi modules at startup, building an
in-memory index of classes, dataclass fields, functions, and methods.
Provides keyword search and exact-name lookup for LLM agents.

Modules are auto-discovered via ``pkgutil.walk_packages`` rather than a
hardcoded list, so the index automatically picks up new modules when the
installed oumi version changes.
"""

import ast
import dataclasses
import heapq
import importlib
import inspect
import logging
import pkgutil
import re
import textwrap
import threading
from collections import defaultdict
from typing import Any

from oumi.mcp.constants import (
    DOCS_MAX_METHODS_PER_CLASS,
    DOCS_MAX_RESULTS,
    DOCS_MODULE_DENYLIST,
    DOCS_MODULE_DENYLIST_PREFIXES,
    DOCS_MODULE_DESCRIPTIONS,
)
from oumi.mcp.models import (
    DocEntry,
    DocsSearchResponse,
    DocstringSection,
    FieldDoc,
    ListModulesResponse,
    ModuleInfo,
)

logger = logging.getLogger(__name__)


def _get_oumi_version() -> str:
    """Return the installed oumi package version, or "unknown"."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("oumi")
    except Exception:
        return "unknown"


_index: list[DocEntry] = []
_module_info: list[ModuleInfo] = []
_qualified_name_map: dict[str, list[DocEntry]] = {}
_name_lower_map: dict[str, list[DocEntry]] = {}
_search_blobs: list[dict[str, Any]] = []
_index_ready = threading.Event()
_index_lock = threading.Lock()

_SECTION_RE = re.compile(
    r"^(Args|Arguments|Returns?|Raises?|Yields?|Examples?|Notes?|Attributes?|See Also|References?|Todo|Warnings?)\s*:",
    re.MULTILINE,
)


def parse_docstring(
    docstring: str | None,
) -> tuple[str, list[DocstringSection]]:
    """Parse a docstring into a summary and sections.

    Args:
        docstring: Raw docstring text (may be None).

    Returns:
        Tuple of (summary_text, list_of_sections).
    """
    if not docstring:
        return ("", [])

    text = textwrap.dedent(docstring).strip()
    if not text:
        return ("", [])

    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return (text, [])

    summary = text[: matches[0].start()].strip()
    sections: list[DocstringSection] = []

    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"name": name, "content": content})

    return (summary, sections)


def _extract_field_docstrings(cls: type) -> dict[str, str]:
    """Extract per-field docstrings from a class using AST analysis.

    Oumi documents dataclass fields with string literals placed directly
    after the field assignment.  ``inspect.getdoc()`` does not capture these,
    so we parse the source with AST and look for ``Expr(Constant(str))``
    nodes following ``AnnAssign`` nodes.

    Args:
        cls: The class to inspect.

    Returns:
        Mapping of field_name -> docstring_text.
    """
    result: dict[str, str] = {}
    try:
        source = inspect.getsource(cls)
    except (OSError, TypeError):
        return result

    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return result

    class_def = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef)),
        None,
    )
    if class_def is None:
        return result

    body = class_def.body
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.AnnAssign):
            continue
        target = stmt.target
        if not isinstance(target, ast.Name):
            continue
        field_name = target.id
        if i + 1 < len(body):
            next_stmt = body[i + 1]
            if isinstance(next_stmt, ast.Expr) and isinstance(
                next_stmt.value, ast.Constant
            ):
                val = next_stmt.value.value
                if isinstance(val, str):
                    result[field_name] = val.strip()

    return result


def _build_field_docs(cls: type) -> list[FieldDoc]:
    """Build documentation for all dataclass fields of *cls*.

    Combines ``dataclasses.fields()`` metadata with AST-extracted docstrings.
    """
    if not dataclasses.is_dataclass(cls):
        return []

    ast_docs = _extract_field_docstrings(cls)
    fields: list[FieldDoc] = []

    for f in dataclasses.fields(cls):
        type_str = ""
        if f.type is not None:
            type_str = (
                getattr(f.type, "__name__", None)
                or getattr(f.type, "_name", None)
                or str(f.type)
            )

        default = ""
        if f.default is not dataclasses.MISSING:
            default = repr(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            factory_name = getattr(
                f.default_factory, "__name__", repr(f.default_factory)
            )
            default = f"{factory_name}()"

        fields.append(
            {
                "name": f.name,
                "type_str": type_str,
                "description": ast_docs.get(f.name, ""),
                "default": default,
            }
        )

    return fields


def _safe_signature(obj: Any) -> str:
    """Return the signature string for *obj*, or "" on failure."""
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return ""


def _get_real_docstring(cls: type) -> str | None:
    """Get the real docstring for a class, avoiding auto-generated repr."""
    doc = cls.__doc__
    if not doc:
        return None
    if dataclasses.is_dataclass(cls) and doc.startswith(f"{cls.__name__}("):
        return None
    return doc


def _index_class(cls: type, module_path: str) -> list[DocEntry]:
    """Index a class and its public methods.

    Args:
        cls: The class to index.
        module_path: Fully qualified module path.

    Returns:
        List of DocEntry dicts (one for the class, plus one per public method).
    """
    entries: list[DocEntry] = []
    qualified = f"{module_path}.{cls.__name__}"

    docstring = _get_real_docstring(cls)
    summary, sections = parse_docstring(docstring)
    is_dc = dataclasses.is_dataclass(cls)
    kind = "dataclass" if is_dc else "class"

    fields = _build_field_docs(cls) if is_dc else []

    entries.append(
        {
            "qualified_name": qualified,
            "name": cls.__name__,
            "kind": kind,
            "module": module_path,
            "summary": summary,
            "sections": sections,
            "fields": fields,
            "signature": _safe_signature(cls),
            "parent_class": "",
        }
    )

    method_count = 0
    for attr_name in sorted(dir(cls)):
        if attr_name.startswith("_"):
            continue
        if method_count >= DOCS_MAX_METHODS_PER_CLASS:
            break

        try:
            attr = getattr(cls, attr_name)
        except Exception:
            continue

        if not (inspect.isfunction(attr) or inspect.ismethod(attr)):
            continue

        method_doc = inspect.getdoc(attr)
        m_summary, m_sections = parse_docstring(method_doc)

        entries.append(
            {
                "qualified_name": f"{qualified}.{attr_name}",
                "name": attr_name,
                "kind": "method",
                "module": module_path,
                "summary": m_summary,
                "sections": m_sections,
                "fields": [],
                "signature": _safe_signature(attr),
                "parent_class": cls.__name__,
            }
        )
        method_count += 1

    return entries


def _index_module(
    module_path: str, description: str
) -> tuple[list[DocEntry], ModuleInfo]:
    """Import and index a single module.

    Args:
        module_path: Fully qualified module path.
        description: Human-readable description.

    Returns:
        Tuple of (entries, module_info).
    """
    entries: list[DocEntry] = []
    class_names: list[str] = []
    function_count = 0

    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        logger.warning("Failed to import %s: %s", module_path, exc)
        info: ModuleInfo = {
            "module": module_path,
            "description": description,
            "class_count": 0,
            "function_count": 0,
            "class_names": [],
        }
        return (entries, info)

    public_exports = set(getattr(mod, "__all__", []))

    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue

        try:
            obj = getattr(mod, name)
        except Exception:
            continue
        obj_module = getattr(obj, "__module__", None)
        if obj_module and obj_module != module_path:
            is_explicit_reexport = name in public_exports and obj_module.startswith(
                f"{module_path}."
            )
            if not is_explicit_reexport:
                continue

        if inspect.isclass(obj):
            class_entries = _index_class(obj, module_path)
            entries.extend(class_entries)
            class_names.append(name)

        elif inspect.isfunction(obj):
            doc = inspect.getdoc(obj)
            summary, sections = parse_docstring(doc)
            entries.append(
                {
                    "qualified_name": f"{module_path}.{name}",
                    "name": name,
                    "kind": "function",
                    "module": module_path,
                    "summary": summary,
                    "sections": sections,
                    "fields": [],
                    "signature": _safe_signature(obj),
                    "parent_class": "",
                }
            )
            function_count += 1

    info = {
        "module": module_path,
        "description": description,
        "class_count": len(class_names),
        "function_count": function_count,
        "class_names": class_names,
    }

    return (entries, info)


def discover_oumi_modules() -> list[tuple[str, str]]:
    """Auto-discover all public oumi modules and pair with descriptions.

    Uses ``pkgutil.walk_packages`` to find every importable ``oumi.*``
    submodule, filters out private/test/denylisted modules, and merges
    curated descriptions from ``DOCS_MODULE_DESCRIPTIONS``.

    Returns:
        List of ``(module_path, description)`` tuples.
    """
    try:
        import oumi
    except ImportError:
        logger.warning("oumi package not importable; falling back to curated list")
        return list(DOCS_MODULE_DESCRIPTIONS.items())

    modules: list[tuple[str, str]] = []
    seen: set[str] = set()

    for _importer, modname, _ispkg in pkgutil.walk_packages(
        oumi.__path__, prefix="oumi."
    ):
        parts = modname.split(".")
        if any(part.startswith("_") for part in parts[1:]):
            continue
        if ".tests." in modname or modname.endswith(".tests"):
            continue
        if modname in DOCS_MODULE_DENYLIST:
            continue
        if any(modname.startswith(prefix) for prefix in DOCS_MODULE_DENYLIST_PREFIXES):
            continue
        if modname in seen:
            continue
        seen.add(modname)

        description = DOCS_MODULE_DESCRIPTIONS.get(modname, "")
        modules.append((modname, description))

    if not modules:
        logger.warning("Auto-discovery found 0 modules; falling back to curated list")
        return list(DOCS_MODULE_DESCRIPTIONS.items())

    logger.info("Auto-discovered %d oumi modules", len(modules))
    return sorted(modules, key=lambda t: t[0])


def _module_docstring_summary(mod: object) -> str:
    """Extract the first sentence of a module's docstring."""
    doc = getattr(mod, "__doc__", None)
    if not doc:
        return ""
    first_line = doc.strip().split("\n")[0].strip().rstrip(".")
    if len(first_line) > 120:
        first_line = first_line[:117] + "..."
    return first_line


def build_index() -> None:
    """Build the full documentation index by auto-discovering oumi modules.

    Iterates over all discovered modules, imports them, and builds the index.
    Sets ``_index_ready`` when complete. Safe to call from any thread.
    """
    global _index, _module_info, _qualified_name_map, _name_lower_map, _search_blobs

    all_entries: list[DocEntry] = []
    all_info: list[ModuleInfo] = []

    discovered_modules = discover_oumi_modules()

    for module_path, description in discovered_modules:
        logger.info("Indexing docs: %s", module_path)
        try:
            entries, info = _index_module(module_path, description)
            if not description and info["class_count"] + info["function_count"] > 0:
                try:
                    mod = importlib.import_module(module_path)
                    auto_desc = _module_docstring_summary(mod)
                    if auto_desc:
                        info["description"] = auto_desc
                except Exception:
                    pass
            if info["class_count"] + info["function_count"] == 0:
                continue
            all_entries.extend(entries)
            all_info.append(info)
        except Exception as exc:
            logger.error("Error indexing %s: %s", module_path, exc, exc_info=True)

    qualified_name_map: dict[str, list[DocEntry]] = defaultdict(list)
    name_lower_map: dict[str, list[DocEntry]] = defaultdict(list)
    search_blobs: list[dict[str, Any]] = []
    for entry in all_entries:
        qualified_name_map[entry["qualified_name"]].append(entry)
        name_lower_map[entry["name"].lower()].append(entry)
        search_blobs.append(
            {
                "entry": entry,
                "name_lower": entry["name"].lower(),
                "qual_lower": entry["qualified_name"].lower(),
                "field_names_lower": [f["name"].lower() for f in entry["fields"]],
                "summary_lower": entry["summary"].lower(),
                "section_contents_lower": [
                    section["content"].lower() for section in entry["sections"]
                ],
                "module_lower": entry["module"].lower(),
                "kind_lower": entry["kind"].lower(),
                "is_class_like": entry["kind"] in ("class", "dataclass"),
            }
        )

    with _index_lock:
        _index = all_entries
        _module_info = all_info
        _qualified_name_map = dict(qualified_name_map)
        _name_lower_map = dict(name_lower_map)
        _search_blobs = search_blobs

    _index_ready.set()
    logger.info(
        "Documentation index ready: %d entries from %d modules",
        len(all_entries),
        len(all_info),
    )


def start_background_indexing() -> threading.Thread:
    """Spawn a daemon thread to build the documentation index.

    Returns:
        The started daemon thread.
    """
    t = threading.Thread(target=build_index, name="docs-indexer", daemon=True)
    t.start()
    return t


def search_docs(
    query: list[str],
    module: str = "",
    kind: str = "",
    limit: int = DOCS_MAX_RESULTS,
    summarize: bool = False,
    examples: bool | None = None,
) -> DocsSearchResponse:
    """Search the documentation index.

    Three-tier matching:
    1. Exact qualified name match.
    2. Exact short name match (case-insensitive).
    3. Scored relevance search across name, fields, summary, and sections.

    Args:
        query: Search terms (names, keywords, or qualified paths). Supports
            multi-keyword search in a single call.
        module: Optional module prefix filter.
        kind: Optional kind filter ("class", "dataclass", "function", "method").
        limit: Maximum results to return.
        summarize: If True, return compact entries with only summary-centric
            content (empty fields/sections).
        examples: Reserved for future filtering behavior.

    Returns:
        DocsSearchResponse with matching entries.
    """

    def _to_summary_entry(entry: DocEntry) -> DocEntry:
        return {
            "qualified_name": entry["qualified_name"],
            "name": entry["name"],
            "kind": entry["kind"],
            "module": entry["module"],
            "summary": entry["summary"],
            "sections": [],
            "fields": [],
            "signature": entry["signature"],
            "parent_class": entry["parent_class"],
        }

    is_ready = _index_ready.is_set()
    query_terms = [q.strip() for q in query if q.strip()]
    query_terms_lower = [q.lower() for q in query_terms]
    module_lower = module.strip().lower()
    kind_lower = kind.strip().lower()
    error_msg = ""
    results: list[DocEntry] = []
    total_matches = 0

    if limit < 1:
        error_msg = "limit must be >= 1."
    else:
        if limit > DOCS_MAX_RESULTS:
            error_msg = f"limit capped to {DOCS_MAX_RESULTS}."
            limit = DOCS_MAX_RESULTS

        if not query_terms:
            error_msg = "Query cannot be empty."
        else:
            with _index_lock:
                index = list(_index)
                qualified_name_map = dict(_qualified_name_map)
                name_lower_map = {k: list(v) for k, v in _name_lower_map.items()}
                search_blobs = list(_search_blobs)

            if not index:
                error_msg = (
                    "Index is empty." if is_ready else "Index is still building."
                )
            else:

                def _entry_matches_filters(entry: DocEntry) -> bool:
                    if module_lower and not entry["module"].lower().startswith(
                        module_lower
                    ):
                        return False
                    if kind_lower and entry["kind"].lower() != kind_lower:
                        return False
                    return True

                filtered_blobs = search_blobs
                if module_lower:
                    filtered_blobs = [
                        blob
                        for blob in filtered_blobs
                        if blob["module_lower"].startswith(module_lower)
                    ]
                if kind_lower:
                    filtered_blobs = [
                        blob
                        for blob in filtered_blobs
                        if blob["kind_lower"] == kind_lower
                    ]

                exact_qual: list[DocEntry] = []
                seen_qual: set[str] = set()
                for term in query_terms:
                    for entry in qualified_name_map.get(term, []):
                        if not _entry_matches_filters(entry):
                            continue
                        entry_key = entry["qualified_name"]
                        if entry_key in seen_qual:
                            continue
                        seen_qual.add(entry_key)
                        exact_qual.append(entry)
                if exact_qual:
                    total_matches = len(exact_qual)
                    results = exact_qual[:limit]
                else:
                    exact_name: list[DocEntry] = []
                    seen_name: set[str] = set()
                    for term_lower in query_terms_lower:
                        for entry in name_lower_map.get(term_lower, []):
                            if not _entry_matches_filters(entry):
                                continue
                            entry_key = entry["qualified_name"]
                            if entry_key in seen_name:
                                continue
                            seen_name.add(entry_key)
                            exact_name.append(entry)
                    if exact_name:
                        total_matches = len(exact_name)
                        results = exact_name[:limit]
                    else:
                        scored: list[tuple[float, int, DocEntry]] = []
                        for idx, blob in enumerate(filtered_blobs):
                            term_scores = [
                                _score_entry(blob, term_lower)
                                for term_lower in query_terms_lower
                            ]
                            matched_scores = [s for s in term_scores if s > 0]
                            if matched_scores:
                                total_matches += 1
                                score = sum(matched_scores)
                                scored.append((score, -idx, blob["entry"]))

                        top_results = heapq.nlargest(limit, scored)
                        results = [entry for _, __, entry in top_results]

    if summarize:
        results = [_to_summary_entry(entry) for entry in results]

    return {
        "results": results,
        "query": query_terms,
        "total_matches": total_matches,
        "index_ready": is_ready,
        "oumi_version": _get_oumi_version(),
        "error": error_msg,
    }


def _score_entry(blob: dict[str, Any], query_lower: str) -> float:
    """Score a precomputed search blob against a lowered query string."""
    score = 0.0
    name_lower = blob["name_lower"]
    qual_lower = blob["qual_lower"]

    if query_lower in name_lower:
        score += 100 if name_lower.startswith(query_lower) else 50
    elif query_lower in qual_lower:
        score += 30

    for field_name_lower in blob["field_names_lower"]:
        if query_lower in field_name_lower:
            score += 30 if field_name_lower == query_lower else 10

    if query_lower in blob["summary_lower"]:
        score += 5

    for section_content_lower in blob["section_contents_lower"]:
        if query_lower in section_content_lower:
            score += 2

    if score > 0 and blob["is_class_like"]:
        score += 10

    return score


def get_module_list() -> ListModulesResponse:
    """Return summaries of all indexed modules.

    Returns:
        ListModulesResponse with module info and total entry count.
    """
    is_ready = _index_ready.is_set()

    with _index_lock:
        info = list(_module_info)
        total = len(_index)

    return {
        "modules": info,
        "total_entries": total,
        "index_ready": is_ready,
        "oumi_version": _get_oumi_version(),
    }
