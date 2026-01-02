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

"""Telemetry manager for Oumi usage analytics using PostHog."""

from __future__ import annotations

import atexit
import json
import os
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import posthog

from oumi.utils.logging import get_logger
from oumi.utils.version_utils import get_oumi_version

OPT_OUT_MESSAGE = (
    "Anonymous analytics are enabled by default to help improve Oumi. "
    "Set DO_NOT_TRACK=1 to disable. "
    "See https://docs.oumi.ai/latest/about/telemetry.html for details."
)


@dataclass
class _TelemetryConfig:
    """Configuration stored in ~/.oumi/telemetry.json."""

    analytics_enabled: bool = True
    install_id: str | None = None


class TelemetryManager:
    """Singleton manager for telemetry collection via PostHog.

    Analytics is enabled by default but can be disabled by setting DO_NOT_TRACK=1.

    Example:
        telemetry = TelemetryManager.get_instance()
        with telemetry.capture_operation("my-event"):
            do_work()  # Timing, success/failure, and exceptions captured automatically
    """

    _instance: TelemetryManager | None = None
    _POSTHOG_HOST = "https://us.i.posthog.com"
    _POSTHOG_API_KEY = "phc_k5jx6NF3FXzWjDWWcj8hKF6RpPHfuimXUz7i3DZxoDZ"
    _OUMI_DIR = Path("~/.oumi").expanduser()
    _TELEMETRY_CONFIG_FILE = _OUMI_DIR / "telemetry.json"

    def __init__(self) -> None:
        """Initialize the telemetry manager. Use get_instance() instead."""
        self._logger = get_logger("oumi.telemetry")
        self._client: posthog.Posthog | None = None
        self._distinct_id: str | None = None
        self._system_info_sent: bool = False

        enabled, install_id = self._init_telemetry()
        if not enabled:
            return

        self._distinct_id = install_id
        self._client = posthog.Posthog(
            project_api_key=self._POSTHOG_API_KEY,
            host=self._POSTHOG_HOST,
            enable_exception_autocapture=True,
            disable_geoip=True,
            super_properties={
                "oumi_version": get_oumi_version(),
                "run_id": str(uuid.uuid4()),
            },
        )
        atexit.register(self._shutdown)

    @classmethod
    def get_instance(cls) -> TelemetryManager:
        """Get the singleton instance of TelemetryManager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self._client is not None

    def _init_telemetry(self) -> tuple[bool, str | None]:
        """Initialize telemetry settings and return (enabled, install_id)."""
        # Only enable telemetry on rank 0 (check env vars to avoid heavy torch import)
        rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", 0)))
        if rank != 0:
            return False, None

        # Check opt-out environment variable
        if os.getenv("DO_NOT_TRACK", "").lower() in ("1", "true"):
            return False, None

        config = self._read_config()

        # First run - show message, create install_id, and save
        if config.install_id is None:
            print(OPT_OUT_MESSAGE)
            config.install_id = str(uuid.uuid4())
            config.analytics_enabled = True
            self._write_config(config)

        return config.analytics_enabled, config.install_id

    @classmethod
    def _read_config(cls) -> _TelemetryConfig:
        """Read telemetry config from ~/.oumi/telemetry.json."""
        if not cls._TELEMETRY_CONFIG_FILE.exists():
            return _TelemetryConfig()
        try:
            with open(cls._TELEMETRY_CONFIG_FILE) as f:
                data = json.load(f)
                return _TelemetryConfig(
                    analytics_enabled=data.get("analytics_enabled", True),
                    install_id=data.get("install_id"),
                )
        except Exception:
            return _TelemetryConfig()

    @classmethod
    def _write_config(cls, config: _TelemetryConfig) -> None:
        """Write telemetry config to ~/.oumi/telemetry.json."""
        try:
            cls._OUMI_DIR.mkdir(parents=True, exist_ok=True)
            with open(cls._TELEMETRY_CONFIG_FILE, "w") as f:
                json.dump(asdict(config), f, indent=2)
        except OSError:
            pass

    @contextmanager
    def _context(self) -> Generator[None]:
        """Create a PostHog context for automatic exception capture."""
        if not self._client or not self._distinct_id:
            yield
            return

        with posthog.new_context(capture_exceptions=True, client=self._client):
            posthog.identify_context(self._distinct_id)
            yield

    def tags(self, **kwargs: Any) -> None:
        """Add multiple tags to the current context."""
        for key, value in kwargs.items():
            posthog.tag(key, value)

    def _capture(self, event: str, properties: dict[str, Any] | None = None) -> None:
        """Capture a telemetry event."""
        if not self._client or not self._distinct_id:
            return

        try:
            props = dict(properties or {})

            # Attach system info as person properties on first event
            if not self._system_info_sent:
                try:
                    from oumi.utils.system_info import get_system_info

                    props["$set"] = get_system_info()
                    self._system_info_sent = True
                except Exception as e:
                    self._logger.debug(f"Failed to collect system info: {e}")

            self._client.capture(
                distinct_id=self._distinct_id,
                event=event,
                properties=props,
            )
        except Exception as e:
            self._logger.debug(f"Failed to capture event: {e}")

    @contextmanager
    def capture_operation(
        self, event: str, properties: dict[str, Any] | None = None
    ) -> Generator[None]:
        """Context manager that captures timing, success/failure, and exceptions.

        No-op when telemetry is disabled.

        Args:
            event: The event name
            properties: Initial properties to include with the event.

        Example:
            with telemetry.capture_operation("cli-train"):
                telemetry.tag("trainer_type", "TRL_SFT")
                run_training()
        """
        if not self._client:
            yield
            return

        start_time = time.time()
        success = True
        exit_code: int | None = None
        error_type: str | None = None

        with self._context():
            try:
                yield
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else None
                success = exit_code is None or exit_code == 0
                raise
            except BaseException as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                props = {
                    **(properties or {}),
                    "success": success,
                    "duration_seconds": round(time.time() - start_time, 2),
                }
                if exit_code is not None:
                    props["exit_code"] = exit_code
                if error_type:
                    props["error_type"] = error_type
                self._capture(event, props)

    def _shutdown(self) -> None:
        if self._client:
            try:
                self._client.shutdown()
            except Exception as e:
                self._logger.debug(f"Failed to shutdown PostHog: {e}")
