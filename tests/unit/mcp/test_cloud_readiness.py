# pyright: reportAttributeAccessIssue=false, reportReturnType=false, reportArgumentType=false
"""Tests for SkyPilot cloud readiness detection across API versions."""

import enum
import types
from contextlib import contextmanager
from unittest.mock import patch

import oumi.mcp.preflight_service as _server_mod
from oumi.mcp.preflight_service import check_cloud_readiness


class _FakeCloud:
    def __init__(self, name: str) -> None:
        self._name = name

    def __str__(self) -> str:
        return self._name


def _build_fake_sky_modules(
    *,
    require_capability: bool,
    enabled_values: list[object],
    include_cloud_capability: bool = True,
    include_check_capability: bool = True,
    check_capability_values: list[str] | None = None,
) -> dict[str, object]:
    sky_mod = types.ModuleType("sky")
    check_mod = types.ModuleType("sky.check")

    def get_cached_enabled_clouds_or_refresh(*args, **kwargs):  # noqa: ANN002, ANN003
        if require_capability and not args:
            raise TypeError(
                "get_cached_enabled_clouds_or_refresh() missing 1"
                " required positional argument: 'capability'"
            )
        return enabled_values

    check_mod.get_cached_enabled_clouds_or_refresh = (
        get_cached_enabled_clouds_or_refresh
    )

    if include_check_capability:
        result_values = check_capability_values or []

        def check_capability(capability, quiet=True, clouds=None):  # noqa: ANN001, ANN202
            del capability, quiet
            if clouds:
                cloud_names = [name.upper() for name in clouds]
                return {
                    "default": [name for name in result_values if name in cloud_names]
                }
            return {"default": result_values}

        check_mod.check_capability = check_capability

    modules: dict[str, object] = {"sky": sky_mod, "sky.check": check_mod}
    sky_mod.check = check_mod

    cloud_capability = None
    if include_cloud_capability:
        clouds_mod = types.ModuleType("sky.clouds")
        cloud_mod = types.ModuleType("sky.clouds.cloud")

        class CloudCapability(enum.Enum):
            COMPUTE = "compute"

        cloud_mod.CloudCapability = CloudCapability
        cloud_capability = CloudCapability
        clouds_mod.cloud = cloud_mod
        sky_mod.clouds = clouds_mod
        modules["sky.clouds"] = clouds_mod
        modules["sky.clouds.cloud"] = cloud_mod

    return modules, sky_mod, check_mod, cloud_capability


@contextmanager
def _patch_sky(modules, sky_mod, check_mod, cloud_capability):
    with (
        patch.object(_server_mod, "sky", sky_mod),
        patch.object(_server_mod, "sky_check", check_mod),
        patch.object(_server_mod, "CloudCapability", cloud_capability),
    ):
        yield


def test_new_skypilot_api_with_cloud_objects():
    modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
        require_capability=True,
        enabled_values=[_FakeCloud("AWS")],
        include_cloud_capability=True,
        include_check_capability=True,
        check_capability_values=["AWS"],
    )
    with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
        errors, warnings, readiness = check_cloud_readiness(target_cloud="aws")

    assert errors == []
    assert warnings == []
    assert readiness["enabled_clouds"] == ["AWS"]
    assert readiness["target_cloud_ready"]


def test_api_mismatch_warning_non_targeted():
    modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
        require_capability=True,
        enabled_values=["AWS"],
        include_cloud_capability=False,
        include_check_capability=False,
    )
    with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
        errors, warnings, readiness = check_cloud_readiness()

    assert errors == []
    assert readiness["target_cloud_ready"] is None
    assert any("SkyPilot API compatibility issue" in w for w in warnings)


def test_api_mismatch_blocking_targeted():
    modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
        require_capability=True,
        enabled_values=["AWS"],
        include_cloud_capability=False,
        include_check_capability=False,
    )
    with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
        errors, warnings, readiness = check_cloud_readiness(target_cloud="aws")

    assert warnings == []
    assert not readiness["target_cloud_ready"]
    assert any("SkyPilot API compatibility error" in e for e in errors)


def test_string_cloud_names_normalized():
    modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
        require_capability=False,
        enabled_values=["gcp", "AWS"],
        include_cloud_capability=True,
        include_check_capability=True,
    )
    with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
        errors, warnings, readiness = check_cloud_readiness()

    assert errors == []
    assert warnings == []
    assert readiness["enabled_clouds"] == ["AWS", "GCP"]
