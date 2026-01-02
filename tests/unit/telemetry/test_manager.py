import json
from unittest.mock import MagicMock, patch

import pytest

from oumi.telemetry.manager import TelemetryManager


def _reset_manager() -> None:
    TelemetryManager._instance = None


def _patch_telemetry_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    oumi_dir = tmp_path / "oumi"
    config_file = oumi_dir / "telemetry.json"
    monkeypatch.setattr(TelemetryManager, "_OUMI_DIR", oumi_dir)
    monkeypatch.setattr(TelemetryManager, "_TELEMETRY_CONFIG_FILE", config_file)


def _patch_telemetry_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)


def test_config_created_on_first_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _reset_manager()
    _patch_telemetry_paths(monkeypatch, tmp_path)
    _patch_telemetry_runtime(monkeypatch)

    # Mock PostHog to avoid real connections
    with patch("posthog.Posthog", MagicMock()):
        manager = TelemetryManager.get_instance()

    assert manager.enabled is True
    config_path = TelemetryManager._TELEMETRY_CONFIG_FILE
    assert config_path.exists()

    config = json.loads(config_path.read_text())
    assert config["analytics_enabled"] is True
    assert isinstance(config.get("install_id"), str)
    assert config["install_id"]


def test_do_not_track_disables_telemetry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _reset_manager()
    _patch_telemetry_paths(monkeypatch, tmp_path)
    _patch_telemetry_runtime(monkeypatch)

    config_path = TelemetryManager._TELEMETRY_CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"analytics_enabled": True, "install_id": "install-123"})
    )

    monkeypatch.setenv("DO_NOT_TRACK", "1")

    manager = TelemetryManager.get_instance()
    assert manager.enabled is False


def test_config_disables_telemetry(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _reset_manager()
    _patch_telemetry_paths(monkeypatch, tmp_path)
    _patch_telemetry_runtime(monkeypatch)

    config_path = TelemetryManager._TELEMETRY_CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"analytics_enabled": False, "install_id": "install-456"})
    )

    manager = TelemetryManager.get_instance()
    assert manager.enabled is False
