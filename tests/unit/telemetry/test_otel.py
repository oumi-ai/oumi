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

"""Tests for OpenTelemetry integration."""

import os
from dataclasses import dataclass
from unittest import mock

import pytest


def _reset_otel_state():
    """Helper to reset OTel module state between tests."""
    import oumi.telemetry.otel as otel_module

    otel_module._initialized = False
    otel_module._tracer = None
    otel_module._enabled_cache = None


class TestOtelConfiguration:
    """Test OTel configuration from environment variables."""

    def test_otel_disabled_by_default(self):
        """Test that OTel is disabled when env var is not set."""
        from oumi.telemetry.otel import _is_otel_enabled

        _reset_otel_state()
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _is_otel_enabled() is False
        _reset_otel_state()

    def test_otel_disabled_with_false(self):
        """Test that OTel is disabled when env var is false."""
        from oumi.telemetry.otel import _is_otel_enabled

        _reset_otel_state()
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            assert _is_otel_enabled() is False
        _reset_otel_state()

    def test_otel_enabled_with_env_var(self):
        """Test that OTel is enabled when env var is set to true."""
        from oumi.telemetry.otel import _is_otel_enabled

        for value in ("true", "1", "yes", "True", "TRUE"):
            _reset_otel_state()
            with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": value}):
                assert _is_otel_enabled() is True
            _reset_otel_state()

    def test_enabled_check_is_cached(self):
        """Test that the enabled check result is cached."""
        import oumi.telemetry.otel as otel_module

        _reset_otel_state()

        # First call should set the cache
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "true"}):
            result1 = otel_module._is_otel_enabled()
            assert result1 is True
            assert otel_module._enabled_cache is True

            # Changing env var shouldn't affect cached result
            with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
                result2 = otel_module._is_otel_enabled()
                # Should still be True because of cache
                assert result2 is True

        _reset_otel_state()

    def test_parse_headers_empty(self):
        """Test header parsing with empty string."""
        from oumi.telemetry.otel import _parse_headers

        assert _parse_headers("") == {}

    def test_parse_headers_single(self):
        """Test header parsing with single header."""
        from oumi.telemetry.otel import _parse_headers

        assert _parse_headers("key=value") == {"key": "value"}

    def test_parse_headers_multiple(self):
        """Test header parsing with multiple headers."""
        from oumi.telemetry.otel import _parse_headers

        assert _parse_headers("k1=v1,k2=v2") == {"k1": "v1", "k2": "v2"}

    def test_parse_headers_with_equals_in_value(self):
        """Test header parsing when value contains equals sign."""
        from oumi.telemetry.otel import _parse_headers

        assert _parse_headers("key=value=with=equals") == {"key": "value=with=equals"}

    def test_parse_headers_with_spaces(self):
        """Test header parsing strips whitespace."""
        from oumi.telemetry.otel import _parse_headers

        assert _parse_headers(" key = value ") == {"key": "value"}

    def test_get_config_defaults(self):
        """Test default configuration values."""
        from oumi.telemetry.otel import _get_config

        with mock.patch.dict(os.environ, {}, clear=True):
            config = _get_config()
            assert config["endpoint"] is None
            assert config["service_name"] == "oumi"
            assert config["headers"] == {}

    def test_get_config_from_env(self):
        """Test configuration from environment variables."""
        from oumi.telemetry.otel import _get_config

        env = {
            "OUMI_OTEL_ENDPOINT": "http://localhost:4317",
            "OUMI_OTEL_SERVICE_NAME": "test-service",
            "OUMI_OTEL_HEADERS": "auth=token123",
        }
        with mock.patch.dict(os.environ, env):
            config = _get_config()
            assert config["endpoint"] == "http://localhost:4317"
            assert config["service_name"] == "test-service"
            assert config["headers"] == {"auth": "token123"}


class TestOtelSpan:
    """Test OTel span context manager."""

    def test_span_noop_when_disabled(self):
        """Test that span is no-op when OTel is disabled."""
        from oumi.telemetry.otel import otel_span

        _reset_otel_state()
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            with otel_span("test.operation") as span:
                assert span is None
        _reset_otel_state()

    def test_span_captures_exception(self):
        """Test that span works correctly when exception is raised."""
        from oumi.telemetry.otel import otel_span

        _reset_otel_state()
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            with pytest.raises(ValueError, match="test error"):
                with otel_span("test.operation"):
                    raise ValueError("test error")
        _reset_otel_state()


class TestTrackFeatureDecorator:
    """Test the track_feature decorator."""

    def test_decorator_preserves_function_behavior(self):
        """Test that decorator doesn't change function behavior."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        @track_feature("test_feature")
        def my_function(x: int) -> int:
            return x * 2

        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            assert my_function(5) == 10

        _reset_otel_state()

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        from oumi.telemetry.otel import track_feature

        @track_feature("test")
        def documented_function():
            """This is a docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == """This is a docstring."""

    def test_decorator_extracts_model_name(self):
        """Test that decorator can extract model name from config."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        @dataclass
        class MockConfig:
            model_name: str

        @track_feature(
            "test",
            get_model_name=lambda c: c.model_name,
        )
        def my_function(config):
            return config.model_name

        config = MockConfig(model_name="gpt2")
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            assert my_function(config) == "gpt2"

        _reset_otel_state()

    def test_decorator_extracts_trainer_type(self):
        """Test that decorator can extract trainer type from config."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        @dataclass
        class MockTrainingParams:
            trainer_type: str

        @dataclass
        class MockConfig:
            training: MockTrainingParams

        @track_feature(
            "train",
            get_trainer_type=lambda c: c.training.trainer_type,
        )
        def my_function(config):
            return config.training.trainer_type

        config = MockConfig(training=MockTrainingParams(trainer_type="TRL_SFT"))
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            assert my_function(config) == "TRL_SFT"

        _reset_otel_state()

    def test_decorator_extracts_inference_engine(self):
        """Test that decorator can extract inference engine from config."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        @dataclass
        class MockConfig:
            engine: str

        @track_feature(
            "infer",
            get_inference_engine=lambda c: c.engine,
        )
        def my_function(config):
            return config.engine

        config = MockConfig(engine="VLLM")
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            assert my_function(config) == "VLLM"

        _reset_otel_state()

    def test_decorator_handles_extraction_error(self):
        """Test that decorator handles extraction errors gracefully."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        @track_feature(
            "test",
            get_model_name=lambda c: c.nonexistent_field,  # Will raise AttributeError
        )
        def my_function(config):
            return "success"

        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            # Should not raise even though extraction fails
            result = my_function({"key": "value"})
            assert result == "success"

        _reset_otel_state()

    def test_decorator_propagates_exceptions(self):
        """Test that decorator re-raises exceptions from decorated function."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        @track_feature("test")
        def failing_function():
            raise RuntimeError("Function failed")

        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            with pytest.raises(RuntimeError, match="Function failed"):
                failing_function()

        _reset_otel_state()

    def test_early_bailout_when_disabled(self):
        """Test that decorator has early bailout when OTel is disabled."""
        from oumi.telemetry.otel import track_feature

        _reset_otel_state()

        call_count = 0

        @track_feature(
            "test",
            get_model_name=lambda c: "should_not_be_called",
        )
        def my_function():
            nonlocal call_count
            call_count += 1
            return "result"

        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            result = my_function()
            assert result == "result"
            assert call_count == 1

        _reset_otel_state()


class TestInitialization:
    """Test OTel initialization behavior."""

    def test_initialization_returns_false_when_disabled(self):
        """Test initialization returns False when OTel is disabled."""
        from oumi.telemetry.otel import _initialize_otel

        _reset_otel_state()
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            result = _initialize_otel()
            assert result is False
        _reset_otel_state()

    def test_get_tracer_returns_none_when_disabled(self):
        """Test get_tracer returns None when OTel is disabled."""
        from oumi.telemetry.otel import get_tracer

        _reset_otel_state()
        with mock.patch.dict(os.environ, {"OUMI_OTEL_ENABLED": "false"}):
            tracer = get_tracer()
            assert tracer is None
        _reset_otel_state()


class TestShutdown:
    """Test OTel shutdown behavior."""

    def test_shutdown_when_not_initialized(self):
        """Test shutdown is safe when not initialized."""
        from oumi.telemetry.otel import shutdown

        _reset_otel_state()
        # Should not raise
        shutdown()

    def test_shutdown_resets_cache(self):
        """Test that shutdown resets the enabled cache when tracer is set."""
        import oumi.telemetry.otel as otel_module
        from oumi.telemetry.otel import shutdown

        _reset_otel_state()

        # Set the cache and a mock tracer
        otel_module._enabled_cache = True
        otel_module._initialized = True
        otel_module._tracer = "mock_tracer"  # Set a non-None value

        # Mock the opentelemetry.trace import inside shutdown
        mock_provider = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules",
            {
                "opentelemetry": mock.MagicMock(),
                "opentelemetry.trace": mock.MagicMock(),
            },
        ):
            import sys

            sys.modules[
                "opentelemetry.trace"
            ].get_tracer_provider.return_value = mock_provider
            shutdown()

        assert otel_module._enabled_cache is None
        assert otel_module._initialized is False
        assert otel_module._tracer is None
