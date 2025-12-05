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

"""OpenTelemetry integration for Oumi feature/API usage tracking.

This module provides a minimal OpenTelemetry setup for tracking feature usage
at main entry points. It operates independently from the existing TelemetryTracker.

Configuration via environment variables:
- OUMI_OTEL_ENABLED: Set to "true" to enable (default: "false")
- OUMI_OTEL_ENDPOINT: OTLP endpoint URL (default: None, uses console exporter)
- OUMI_OTEL_SERVICE_NAME: Service name (default: "oumi")
- OUMI_OTEL_HEADERS: Comma-separated key=value pairs for headers
"""

import functools
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar

from oumi.utils.logging import logger

# Type variable for decorator
F = TypeVar("F", bound=Callable[..., Any])

# Global state
_tracer = None
_initialized = False
_enabled_cache: Optional[bool] = None


def _is_otel_enabled() -> bool:
    """Check if OpenTelemetry is enabled via environment variable.

    The result is cached after the first call for performance.
    """
    global _enabled_cache
    if _enabled_cache is None:
        _enabled_cache = os.environ.get("OUMI_OTEL_ENABLED", "false").lower() in (
            "true",
            "1",
            "yes",
        )
    return _enabled_cache


def _get_config() -> dict[str, Any]:
    """Get OTel configuration from environment variables."""
    return {
        "endpoint": os.environ.get("OUMI_OTEL_ENDPOINT"),
        "service_name": os.environ.get("OUMI_OTEL_SERVICE_NAME", "oumi"),
        "headers": _parse_headers(os.environ.get("OUMI_OTEL_HEADERS", "")),
    }


def _parse_headers(headers_str: str) -> dict[str, str]:
    """Parse comma-separated key=value headers string."""
    if not headers_str:
        return {}
    headers = {}
    for pair in headers_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers


def _initialize_otel() -> bool:
    """Initialize OpenTelemetry tracer provider.

    Returns:
        True if initialization succeeded, False otherwise.
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer is not None

    _initialized = True

    if not _is_otel_enabled():
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        config = _get_config()

        # Create resource with service name
        resource = Resource.create({"service.name": config["service_name"]})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure exporter
        if config["endpoint"]:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(
                endpoint=config["endpoint"],
                headers=config["headers"] or None,
            )
        else:
            # Fallback to console exporter for debugging
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            exporter = ConsoleSpanExporter()
            logger.debug("OUMI_OTEL_ENDPOINT not set, using console exporter")

        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _tracer = trace.get_tracer("oumi", "1.0.0")
        logger.debug(f"OpenTelemetry initialized with endpoint: {config['endpoint']}")
        return True

    except ImportError as e:
        logger.debug(f"OpenTelemetry packages not available: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize OpenTelemetry: {e}")
        return False


def get_tracer():
    """Get the OpenTelemetry tracer, initializing if needed.

    Returns:
        The tracer instance, or None if OTel is not available/enabled.
    """
    _initialize_otel()
    return _tracer


@contextmanager
def otel_span(
    name: str,
    attributes: Optional[dict[str, Any]] = None,
):
    """Context manager for creating an OpenTelemetry span.

    Args:
        name: Span name (e.g., "oumi.train", "oumi.infer")
        attributes: Optional attributes to attach to the span

    Yields:
        The span object (or None if OTel is disabled)
    """
    tracer = get_tracer()

    if tracer is None:
        # No-op context when OTel is not available
        yield None
        return

    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    # Convert to string for non-primitive types
                    if isinstance(value, (str, int, float, bool)):
                        span.set_attribute(key, value)
                    else:
                        span.set_attribute(key, str(value))

        start_time = time.perf_counter()
        try:
            yield span
            span.set_attribute("oumi.status", "success")
        except Exception as e:
            span.set_attribute("oumi.status", "error")
            span.set_attribute("oumi.error_type", type(e).__name__)
            span.set_attribute("oumi.error_message", str(e)[:500])
            span.record_exception(e)
            raise
        finally:
            duration = time.perf_counter() - start_time
            span.set_attribute("oumi.duration_seconds", duration)


def track_feature(
    feature_name: str,
    *,
    get_model_name: Optional[Callable[[Any], Optional[str]]] = None,
    get_dataset_name: Optional[Callable[[Any], Optional[str]]] = None,
    get_trainer_type: Optional[Callable[[Any], Optional[str]]] = None,
    get_inference_engine: Optional[Callable[[Any], Optional[str]]] = None,
) -> Callable[[F], F]:
    """Decorator to track feature/API usage.

    Args:
        feature_name: Name of the feature (e.g., "train", "infer")
        get_model_name: Optional callable to extract model name from config arg
        get_dataset_name: Optional callable to extract dataset name from config arg
        get_trainer_type: Optional callable to extract trainer type from config arg
        get_inference_engine: Optional callable to extract inference engine from config

    Returns:
        Decorated function

    Example:
        @track_feature(
            "train",
            get_model_name=lambda config: config.model.model_name,
            get_trainer_type=lambda config: str(config.training.trainer_type.value),
        )
        def train(config: TrainingConfig) -> None:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Fast path: skip all telemetry logic when disabled
            if not _is_otel_enabled():
                return func(*args, **kwargs)

            # Build attributes
            attributes: dict[str, Any] = {
                "oumi.feature": feature_name,
            }

            # Try to extract attributes from first arg (typically config)
            config = args[0] if args else kwargs.get("config")

            if config is not None:
                if get_model_name:
                    try:
                        model_name = get_model_name(config)
                        if model_name:
                            attributes["oumi.model_name"] = model_name
                    except Exception:
                        pass

                if get_dataset_name:
                    try:
                        dataset_name = get_dataset_name(config)
                        if dataset_name:
                            attributes["oumi.dataset_name"] = dataset_name
                    except Exception:
                        pass

                if get_trainer_type:
                    try:
                        trainer_type = get_trainer_type(config)
                        if trainer_type:
                            attributes["oumi.trainer_type"] = trainer_type
                    except Exception:
                        pass

                if get_inference_engine:
                    try:
                        engine = get_inference_engine(config)
                        if engine:
                            attributes["oumi.inference_engine"] = engine
                    except Exception:
                        pass

            with otel_span(f"oumi.{feature_name}", attributes=attributes):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def shutdown():
    """Shutdown the OpenTelemetry tracer provider gracefully."""
    global _tracer, _initialized, _enabled_cache

    if not _initialized or _tracer is None:
        return

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        logger.debug("OpenTelemetry shutdown complete")
    except Exception as e:
        logger.debug(f"Error during OpenTelemetry shutdown: {e}")
    finally:
        _tracer = None
        _initialized = False
        _enabled_cache = None
