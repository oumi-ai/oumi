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

"""Unit tests for the LlamaCppQuantization (GGUF) backend."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.configs import ModelParams, QuantizationConfig
from oumi.core.configs.quantization_config import (
    QuantizationAlgorithm,
    QuantizationBackend,
    QuantizationScheme,
)
from oumi.exceptions import OumiConfigError
from oumi.quantize.base import QuantizationResult
from oumi.quantize.llamacpp import (
    _SCHEME_TO_LLAMACPP_NAME,
    LlamaCppQuantization,
)


def _make_config(
    scheme: QuantizationScheme = QuantizationScheme.Q4_0, **overrides: Any
) -> QuantizationConfig:
    defaults: dict[str, Any] = {
        "model": ModelParams(model_name="test/model"),
        "scheme": scheme,
        "output_path": "test_out",
        "output_format": "gguf",
    }
    defaults.update(overrides)
    return QuantizationConfig(**defaults)


class TestSchemeMetadata:
    def test_backend_identity(self):
        assert LlamaCppQuantization.backend is QuantizationBackend.LLAMACPP

    def test_output_format_is_gguf(self):
        assert LlamaCppQuantization.output_format == "gguf"

    @pytest.mark.parametrize(
        "scheme",
        [
            QuantizationScheme.Q4_0,
            QuantizationScheme.Q4_K_M,
            QuantizationScheme.Q5_K_M,
            QuantizationScheme.Q6_K,
            QuantizationScheme.Q8_0,
        ],
    )
    def test_owns(self, scheme):
        assert LlamaCppQuantization.owns(scheme) is True

    def test_does_not_own_other_schemes(self):
        for s in (
            QuantizationScheme.FP8_DYNAMIC,
            QuantizationScheme.W4A16,
            QuantizationScheme.BNB_NF4,
        ):
            assert LlamaCppQuantization.owns(s) is False

    def test_only_llamacpp_algorithm_allowed(self):
        for s in LlamaCppQuantization.schemes.values():
            assert s.allowed_algorithms == (QuantizationAlgorithm.LLAMACPP,)
            assert s.default_algorithm is QuantizationAlgorithm.LLAMACPP

    def test_calibration_defaults(self):
        s = LlamaCppQuantization.schemes
        assert s[QuantizationScheme.Q4_0].needs_calibration_default is False
        assert s[QuantizationScheme.Q8_0].needs_calibration_default is False
        assert s[QuantizationScheme.Q4_K_M].needs_calibration_default is True
        assert s[QuantizationScheme.Q5_K_M].needs_calibration_default is True
        assert s[QuantizationScheme.Q6_K].needs_calibration_default is True

    def test_scheme_to_llamacpp_name_mapping_complete(self):
        # Every declared scheme must have a name mapping for llama-quantize.
        for scheme in LlamaCppQuantization.schemes:
            assert scheme in _SCHEME_TO_LLAMACPP_NAME
            # Convention: maps to the upstream uppercase form.
            assert _SCHEME_TO_LLAMACPP_NAME[scheme] == scheme.name

    def test_config_rejects_non_llamacpp_algorithm(self):
        with pytest.raises(OumiConfigError, match="not allowed"):
            _make_config(scheme=QuantizationScheme.Q4_0, algorithm="gptq")

    def test_config_rejects_non_gguf_format(self):
        with pytest.raises(OumiConfigError, match="only supports output format"):
            _make_config(scheme=QuantizationScheme.Q4_0, output_format="safetensors")


class TestRequirements:
    def setup_method(self):
        self.quantizer = LlamaCppQuantization()

    def test_missing_binaries_raises(self):
        with patch("oumi.quantize.llamacpp.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="llama.cpp binaries on PATH"):
                self.quantizer.raise_if_requirements_not_met()

    def test_partial_binaries_raises(self):
        # llama-quantize present, llama-imatrix missing.
        def which(name):
            return "/fake/llama-quantize" if name == "llama-quantize" else None

        with patch("oumi.quantize.llamacpp.shutil.which", side_effect=which):
            with pytest.raises(RuntimeError, match="llama-imatrix"):
                self.quantizer.raise_if_requirements_not_met()

    def test_all_binaries_present(self):
        with patch("oumi.quantize.llamacpp.shutil.which", return_value="/fake/bin"):
            self.quantizer.raise_if_requirements_not_met()


class TestQuantizePipeline:
    """Mocked end-to-end: each subprocess invocation is captured."""

    def setup_method(self):
        self.quantizer = LlamaCppQuantization()

    def _patches(self, tmp_path):
        # Single point to wire all mocks consistently.
        return {
            "ensure_tools": patch(
                "oumi.quantize.llamacpp.ensure_llamacpp_python_tools",
                return_value=tmp_path / "llamacpp",
            ),
            "convert_path": patch(
                "oumi.quantize.llamacpp.convert_script_path",
                return_value=tmp_path / "llamacpp" / "convert_hf_to_gguf.py",
            ),
            "gguf_path": patch(
                "oumi.quantize.llamacpp.gguf_py_path",
                return_value=tmp_path / "llamacpp" / "gguf-py",
            ),
            "snapshot": patch(
                "huggingface_hub.snapshot_download",
                return_value=str(tmp_path / "model"),
            ),
            "warn": patch(
                "oumi.quantize.llamacpp.warn_if_local_gpu_below_inference_capability"
            ),
            "writable": patch(
                "oumi.quantize.llamacpp.assert_output_path_writable"
            ),
            "size": patch(
                "oumi.quantize.llamacpp.get_directory_size", return_value=99 * 1024 * 1024
            ),
        }

    def test_q4_0_skips_imatrix(self, tmp_path):
        config = _make_config(
            scheme=QuantizationScheme.Q4_0, output_path=str(tmp_path / "out")
        )
        commands: list[list[str]] = []

        def fake_run(cmd, *, log_prefix, **kw):
            commands.append(list(cmd))

        patches = self._patches(tmp_path)
        with (
            patches["ensure_tools"],
            patches["convert_path"],
            patches["gguf_path"],
            patches["snapshot"] as mock_snap,
            patches["warn"],
            patches["writable"],
            patches["size"],
            patch("oumi.quantize.llamacpp.run_subprocess", side_effect=fake_run),
        ):
            result = self.quantizer.quantize(config)

        # Q4_0 -> 2 subprocesses: convert + quantize. No imatrix.
        assert len(commands) == 2
        assert "convert_hf_to_gguf.py" in " ".join(commands[0])
        assert commands[1][0] == "llama-quantize"
        assert "--imatrix" not in commands[1]
        assert commands[1][-1] == "Q4_0"
        # Snapshot download triggered for HF repo id.
        mock_snap.assert_called_once()

        assert isinstance(result, QuantizationResult)
        assert result.backend is QuantizationBackend.LLAMACPP
        assert result.scheme is QuantizationScheme.Q4_0
        assert result.format_type == "gguf"

    def test_q4_k_m_runs_imatrix(self, tmp_path):
        config = _make_config(
            scheme=QuantizationScheme.Q4_K_M,
            output_path=str(tmp_path / "out"),
            calibration_samples=4,
            max_seq_length=128,
        )
        commands: list[list[str]] = []

        def fake_run(cmd, *, log_prefix, **kw):
            commands.append(list(cmd))

        patches = self._patches(tmp_path)
        with (
            patches["ensure_tools"],
            patches["convert_path"],
            patches["gguf_path"],
            patches["snapshot"],
            patches["warn"],
            patches["writable"],
            patches["size"],
            patch("oumi.quantize.llamacpp.run_subprocess", side_effect=fake_run),
            patch.object(
                self.quantizer, "_write_calibration_corpus", return_value=None
            ),
        ):
            self.quantizer.quantize(config)

        # Q4_K_M -> 3 subprocesses: convert + imatrix + quantize.
        assert len(commands) == 3
        assert commands[1][0] == "llama-imatrix"
        assert "--chunks" in commands[1] and "4" in commands[1]
        assert commands[2][0] == "llama-quantize"
        assert "--imatrix" in commands[2]
        assert commands[2][-1] == "Q4_K_M"

    def test_local_dir_skips_snapshot_download(self, tmp_path):
        local_model = tmp_path / "local-model"
        local_model.mkdir()
        config = _make_config(
            scheme=QuantizationScheme.Q8_0,
            output_path=str(tmp_path / "out"),
            model=ModelParams(model_name=str(local_model)),
        )
        patches = self._patches(tmp_path)
        with (
            patches["ensure_tools"],
            patches["convert_path"],
            patches["gguf_path"],
            patches["snapshot"] as mock_snap,
            patches["warn"],
            patches["writable"],
            patches["size"],
            patch("oumi.quantize.llamacpp.run_subprocess"),
        ):
            self.quantizer.quantize(config)
        # Local directory: no HF download.
        mock_snap.assert_not_called()


class TestBootstrap:
    """Smoke tests on the bootstrap helper. Network-free."""

    def test_cache_root_uses_env_var(self, tmp_path, monkeypatch):
        from oumi.quantize._bootstrap import ENV_LLAMACPP_HOME, cache_root

        monkeypatch.setenv(ENV_LLAMACPP_HOME, str(tmp_path / "custom"))
        assert cache_root() == tmp_path / "custom"

    def test_cache_root_default(self, monkeypatch):
        from oumi.quantize._bootstrap import ENV_LLAMACPP_HOME, cache_root

        monkeypatch.delenv(ENV_LLAMACPP_HOME, raising=False)
        path = cache_root()
        assert path.parts[-3:] == (".cache", "oumi", "llamacpp")

    def test_is_valid_clone_recognizes_required_files(self, tmp_path):
        from oumi.quantize._bootstrap import is_valid_clone

        assert is_valid_clone(tmp_path) is False
        (tmp_path / "convert_hf_to_gguf.py").touch()
        (tmp_path / "gguf-py" / "gguf").mkdir(parents=True)
        (tmp_path / "gguf-py" / "gguf" / "__init__.py").touch()
        assert is_valid_clone(tmp_path) is True

    def test_ensure_returns_existing_clone_without_prompt(self, tmp_path, monkeypatch):
        from oumi.quantize import _bootstrap

        # Pre-populate as a valid clone.
        (tmp_path / "convert_hf_to_gguf.py").touch()
        (tmp_path / "gguf-py" / "gguf").mkdir(parents=True)
        (tmp_path / "gguf-py" / "gguf" / "__init__.py").touch()
        monkeypatch.setenv(_bootstrap.ENV_LLAMACPP_HOME, str(tmp_path))

        with patch("oumi.quantize._bootstrap._do_install") as mock_install:
            result = _bootstrap.ensure_llamacpp_python_tools()
        assert result == tmp_path
        mock_install.assert_not_called()

    def test_ensure_refuses_when_user_declines(self, tmp_path, monkeypatch):
        from oumi.quantize import _bootstrap

        monkeypatch.setenv(_bootstrap.ENV_LLAMACPP_HOME, str(tmp_path / "missing"))
        monkeypatch.delenv(_bootstrap.ENV_AUTO_INSTALL, raising=False)
        # Force non-interactive (no tty) so confirmation returns False.
        with patch("sys.stdin.isatty", return_value=False):
            with patch("oumi.quantize._bootstrap.shutil.which", return_value="/git"):
                with pytest.raises(RuntimeError, match="declined"):
                    _bootstrap.ensure_llamacpp_python_tools()

    def test_ensure_skips_prompt_when_auto_install_set(self, tmp_path, monkeypatch):
        from oumi.quantize import _bootstrap

        monkeypatch.setenv(_bootstrap.ENV_LLAMACPP_HOME, str(tmp_path / "fresh"))
        monkeypatch.setenv(_bootstrap.ENV_AUTO_INSTALL, "1")

        # Simulate _do_install creating the clone files.
        def fake_install(root):
            root.mkdir(parents=True, exist_ok=True)
            (root / "convert_hf_to_gguf.py").touch()
            (root / "gguf-py" / "gguf").mkdir(parents=True)
            (root / "gguf-py" / "gguf" / "__init__.py").touch()

        with patch("oumi.quantize._bootstrap.shutil.which", return_value="/git"):
            with patch(
                "oumi.quantize._bootstrap._do_install", side_effect=fake_install
            ) as mock_install:
                result = _bootstrap.ensure_llamacpp_python_tools()
        mock_install.assert_called_once()
        assert result == tmp_path / "fresh"

    def test_ensure_raises_when_git_missing(self, tmp_path, monkeypatch):
        from oumi.quantize import _bootstrap

        monkeypatch.setenv(_bootstrap.ENV_LLAMACPP_HOME, str(tmp_path / "missing"))
        with patch("oumi.quantize._bootstrap.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="needs `git`"):
                _bootstrap.ensure_llamacpp_python_tools()


class TestRunSubprocess:
    """Sanity test on the shared run_subprocess helper."""

    def test_success(self):
        from oumi.quantize.utils import run_subprocess

        run_subprocess(["true"], log_prefix="test")

    def test_nonzero_exit_includes_tail(self):
        from oumi.quantize.utils import run_subprocess

        with pytest.raises(RuntimeError, match="failed"):
            run_subprocess(["false"], log_prefix="test")

    def test_captures_stdout_in_error(self):
        from oumi.quantize.utils import run_subprocess

        with pytest.raises(RuntimeError, match="hello-from-subprocess"):
            run_subprocess(
                ["sh", "-c", "echo hello-from-subprocess; exit 7"],
                log_prefix="test",
            )
