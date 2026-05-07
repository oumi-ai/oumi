"""Tests for .github/scripts/update_readme_news.py"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".github", "scripts"))

import pytest
from update_readme_news import (
    parse_news_items,
    rewrite_readme_news,
    is_release_in_news,
    format_news_item,
)

SAMPLE_README = """\
## 🔥 News

- [2026/03] Item A
- [2026/02] Item B
- [2025/12] Item C

<details>
<summary>Older updates</summary>

- [2025/09] Item D
- [2025/08] Item E

</details>

## 🔎 About

Some text.
"""

def test_parse_news_items_returns_all_bullets():
    items = parse_news_items(SAMPLE_README)
    assert items == [
        "- [2026/03] Item A",
        "- [2026/02] Item B",
        "- [2025/12] Item C",
        "- [2025/09] Item D",
        "- [2025/08] Item E",
    ]


def test_parse_news_items_no_details_block():
    readme = "## 🔥 News\n\n- [2026/01] Only item\n\n## Next\n"
    items = parse_news_items(readme)
    assert items == ["- [2026/01] Only item"]


def test_parse_news_items_empty_section():
    readme = "## 🔥 News\n\n## Next\n"
    items = parse_news_items(readme)
    assert items == []


def test_is_release_in_news_true():
    readme = "## 🔥 News\n\n- [2025/12] [Oumi v0.6.0 released](https://github.com/oumi-ai/oumi/releases/tag/v0.6.0) with stuff\n\n## Next\n"
    assert is_release_in_news(readme, "v0.6.0") is True


def test_is_release_in_news_false():
    readme = "## 🔥 News\n\n- [2025/12] Some other item\n\n## Next\n"
    assert is_release_in_news(readme, "v0.6.0") is False


def test_is_release_in_news_tag_in_older_block():
    readme = (
        "## 🔥 News\n\n- [2026/01] New item\n\n"
        "<details>\n<summary>Older updates</summary>\n\n"
        "- [2025/09] [Oumi v0.4.0 released](https://github.com/oumi-ai/oumi/releases/tag/v0.4.0)\n\n"
        "</details>\n\n## Next\n"
    )
    assert is_release_in_news(readme, "v0.4.0") is True


def test_format_news_item_with_summary():
    item = format_news_item(
        tag="v0.8",
        url="https://github.com/oumi-ai/oumi/releases/tag/v0.8",
        published_at="2026-05-01T12:00:00Z",
        summary="with new feature X and Y",
    )
    assert item == "- [2026/05] [Oumi v0.8 released](https://github.com/oumi-ai/oumi/releases/tag/v0.8) with new feature X and Y"


def test_format_news_item_empty_summary():
    item = format_news_item(
        tag="v0.9",
        url="https://github.com/oumi-ai/oumi/releases/tag/v0.9",
        published_at="2026-06-15T00:00:00Z",
        summary="",
    )
    assert item == "- [2026/06] [Oumi v0.9 released](https://github.com/oumi-ai/oumi/releases/tag/v0.9)"


def _make_readme(visible_count: int, older_count: int = 0) -> str:
    """Build a synthetic README with given item counts."""
    lines = ["## 🔥 News", ""]
    for i in range(visible_count):
        lines.append(f"- [2026/{visible_count - i:02d}] Item {i + 1}")
    lines.append("")
    if older_count:
        lines += ["<details>", "<summary>Older updates</summary>", ""]
        for j in range(older_count):
            lines.append(f"- [2025/{older_count - j:02d}] Older {j + 1}")
        lines += ["", "</details>", ""]
    lines += ["## 🔎 About", "", "Some text.", ""]
    return "\n".join(lines)


def test_rewrite_prepends_new_item():
    readme = _make_readme(visible_count=3)
    result = rewrite_readme_news(readme, "- [2026/05] New Item")
    items = parse_news_items(result)
    assert items[0] == "- [2026/05] New Item"
    assert len(items) == 4


def test_rewrite_keeps_12_visible_wraps_rest():
    # Start with 12 visible + 5 older = 17 total; adding 1 → 18 total → 12 visible + 6 older
    readme = _make_readme(visible_count=12, older_count=5)
    result = rewrite_readme_news(readme, "- [2026/06] Brand New")
    # Exactly 12 bullets should appear before <details>
    news_start = result.index("## 🔥 News")
    details_start = result.index("<details>", news_start)
    visible_section = result[news_start:details_start]
    visible_bullets = [l for l in visible_section.splitlines() if l.startswith("- [")]
    assert len(visible_bullets) == 12
    # Remaining 6 should be inside details
    items = parse_news_items(result)
    assert len(items) == 18
    assert items[0] == "- [2026/06] Brand New"


def test_rewrite_under_12_no_details_block():
    readme = _make_readme(visible_count=5)
    result = rewrite_readme_news(readme, "- [2026/05] New Item")
    assert "<details>" not in result
    items = parse_news_items(result)
    assert len(items) == 6


def test_rewrite_normalizes_older_summary_label():
    readme = _make_readme(visible_count=12, older_count=3)
    result = rewrite_readme_news(readme, "- [2026/06] Brand New")
    assert "<summary>Older items</summary>" in result


def test_rewrite_preserves_rest_of_readme():
    readme = _make_readme(visible_count=3)
    result = rewrite_readme_news(readme, "- [2026/05] New Item")
    assert "## 🔎 About" in result
    assert "Some text." in result
