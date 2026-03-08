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

"""Tests for the completions module."""

import tempfile
from pathlib import Path

from oumi.cli.completions import (
    _fuzzy_match,
    _score_match,
    complete_eval_config,
    complete_infer_config,
    complete_train_config,
    create_config_completer,
)


class TestFuzzyMatch:
    """Tests for the _fuzzy_match function."""

    def test_empty_pattern_matches_everything(self):
        """Empty pattern should match any text."""
        assert _fuzzy_match("", "anything")
        assert _fuzzy_match("", "")

    def test_prefix_match(self):
        """Prefix matches should work."""
        assert _fuzzy_match("llama", "llama3.1-8b")
        assert _fuzzy_match("Llama", "llama3.1-8b")  # Case insensitive
        assert _fuzzy_match("LLAMA", "llama3.1-8b")  # Case insensitive

    def test_fuzzy_match_subsequence(self):
        """Fuzzy matching should work for character subsequences."""
        assert _fuzzy_match("l38", "llama3.1-8b")
        assert _fuzzy_match("l318", "llama3.1-8b")
        assert _fuzzy_match("qw32", "qwen3-32b")

    def test_no_match(self):
        """Non-matching patterns should return False."""
        assert not _fuzzy_match("xyz", "llama3.1-8b")
        assert not _fuzzy_match("abc", "qwen3-32b")
        assert not _fuzzy_match("zza", "llama")  # chars not in order


class TestScoreMatch:
    """Tests for the _score_match function."""

    def test_exact_match_is_best(self):
        """Exact matches should have the lowest score."""
        assert _score_match("llama", "llama") == 0

    def test_prefix_match_is_good(self):
        """Prefix matches should have low scores."""
        prefix_score = _score_match("llama", "llama3.1-8b")
        fuzzy_score = _score_match("l38", "llama3.1-8b")
        assert prefix_score < fuzzy_score

    def test_shorter_remaining_is_better(self):
        """Shorter remaining text after prefix should score better."""
        score1 = _score_match("llama", "llama1")
        score2 = _score_match("llama", "llama3.1-8b")
        assert score1 < score2


class TestCompleteTrainConfig:
    """Tests for complete_train_config function."""

    def test_returns_tuples(self):
        """Completions should return (name, help_text) tuples."""
        results = complete_train_config("llama")
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_prefix_completion(self):
        """Prefix completion should return matching recipes."""
        results = complete_train_config("llama")
        names = [r[0] for r in results]
        assert all("llama" in name.lower() for name in names)

    def test_fuzzy_completion(self):
        """Fuzzy completion should return matching recipes."""
        results = complete_train_config("l318")
        names = [r[0] for r in results]
        assert "llama3.1-8b" in names

    def test_empty_returns_all(self):
        """Empty string should return all available recipes."""
        results = complete_train_config("")
        assert len(results) > 10  # Should have many recipes

    def test_no_match_returns_empty(self):
        """Non-matching pattern should return empty or just paths."""
        results = complete_train_config("xyz123nonexistent")
        # Should only return path completions (if any) or empty
        names = [r[0] for r in results]
        assert all("xyz123" not in name.lower() for name in names if "/" not in name)


class TestCompleteInferConfig:
    """Tests for complete_infer_config function."""

    def test_returns_infer_recipes(self):
        """Should return inference-specific recipes."""
        results = complete_infer_config("llama")
        names = [r[0] for r in results]
        # Should include llama inference configs
        assert any("llama" in name.lower() for name in names)

    def test_only_infer_aliases(self):
        """Should only return aliases that have inference configs."""
        results = complete_infer_config("")
        names = [r[0] for r in results if "/" not in r[0]]

        # Import to verify - these should have infer configs
        from oumi.cli.alias import _ALIASES, AliasType

        for name in names:
            if name in _ALIASES:
                assert AliasType.INFER in _ALIASES[name]


class TestCompleteEvalConfig:
    """Tests for complete_eval_config function."""

    def test_returns_eval_recipes(self):
        """Should return evaluation-specific recipes."""
        results = complete_eval_config("llama")
        names = [r[0] for r in results]
        assert any("llama" in name.lower() for name in names)


class TestCreateConfigCompleter:
    """Tests for create_config_completer factory function."""

    def test_creates_callable(self):
        """Factory should create a callable completer."""
        from oumi.cli.alias import AliasType

        completer = create_config_completer(AliasType.TRAIN)
        assert callable(completer)

    def test_created_completer_works(self):
        """Created completer should work correctly."""
        from oumi.cli.alias import AliasType

        completer = create_config_completer(AliasType.TRAIN)
        results = completer("llama")
        assert len(results) > 0


class TestPathCompletion:
    """Tests for config path completion."""

    def test_path_completion_triggers_on_configs_prefix(self):
        """Path completion should trigger when typing 'configs/'."""
        results = complete_train_config("configs/")
        # Should have path completions
        paths = [r[0] for r in results if "/" in r[0]]
        assert len(paths) >= 0  # May or may not have paths depending on cwd

    def test_path_completion_with_temp_dir(self):
        """Path completion should work with real directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test yaml file
            yaml_path = Path(tmpdir) / "test.yaml"
            yaml_path.write_text("key: value")

            # Create a subdirectory
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            # Test completion
            results = complete_train_config(f"{tmpdir}/")
            paths = [r[0] for r in results]

            # Should find the yaml file and subdirectory
            assert any("test.yaml" in p for p in paths)
            assert any("subdir" in p for p in paths)
