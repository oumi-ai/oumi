import enum
import types
import unittest
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
                "get_cached_enabled_clouds_or_refresh() missing 1 required positional argument: 'capability'"
            )
        return enabled_values

    check_mod.get_cached_enabled_clouds_or_refresh = get_cached_enabled_clouds_or_refresh

    if include_check_capability:
        result_values = check_capability_values or []

        def check_capability(capability, quiet=True, clouds=None):  # noqa: ANN001, ANN202
            del capability, quiet
            if clouds:
                cloud_names = [name.upper() for name in clouds]
                return {"default": [name for name in result_values if name in cloud_names]}
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
def _patch_sky(modules: dict, sky_mod: object, check_mod: object, cloud_capability: object):
    """Patch module-level sky references in server.py for testing."""
    with (
        patch.object(_server_mod, "sky", sky_mod),
        patch.object(_server_mod, "sky_check", check_mod),
        patch.object(_server_mod, "CloudCapability", cloud_capability),
    ):
        yield


class CloudReadinessCompatibilityTests(unittest.TestCase):
    def test_new_skypilot_api_with_cloud_objects(self) -> None:
        modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
            require_capability=True,
            enabled_values=[_FakeCloud("AWS")],
            include_cloud_capability=True,
            include_check_capability=True,
            check_capability_values=["AWS"],
        )
        with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
            errors, warnings, readiness = check_cloud_readiness(target_cloud="aws")

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(readiness["enabled_clouds"], ["AWS"])
        self.assertTrue(readiness["target_cloud_ready"])

    def test_api_mismatch_is_warning_for_non_targeted_check(self) -> None:
        modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
            require_capability=True,
            enabled_values=["AWS"],
            include_cloud_capability=False,
            include_check_capability=False,
        )
        with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
            errors, warnings, readiness = check_cloud_readiness()

        self.assertEqual(errors, [])
        self.assertEqual(readiness["target_cloud_ready"], None)
        self.assertTrue(
            any("SkyPilot API compatibility issue" in warning for warning in warnings)
        )

    def test_api_mismatch_is_blocking_for_targeted_check(self) -> None:
        modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
            require_capability=True,
            enabled_values=["AWS"],
            include_cloud_capability=False,
            include_check_capability=False,
        )
        with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
            errors, warnings, readiness = check_cloud_readiness(target_cloud="aws")

        self.assertEqual(warnings, [])
        self.assertFalse(readiness["target_cloud_ready"])
        self.assertTrue(
            any("SkyPilot API compatibility error" in error for error in errors)
        )

    def test_string_cloud_names_are_normalized(self) -> None:
        modules, sky_mod, check_mod, cloud_capability = _build_fake_sky_modules(
            require_capability=False,
            enabled_values=["gcp", "AWS"],
            include_cloud_capability=True,
            include_check_capability=True,
        )
        with _patch_sky(modules, sky_mod, check_mod, cloud_capability):
            errors, warnings, readiness = check_cloud_readiness()

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])
        self.assertEqual(readiness["enabled_clouds"], ["AWS", "GCP"])


if __name__ == "__main__":
    unittest.main()
