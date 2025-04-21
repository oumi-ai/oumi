from multiprocessing import Process, set_start_method


def _verify_no_extra_import(extra_module: str):
    """Verifies that extra modules are not imported."""
    import sys

    import oumi.cli.main  # noqa

    assert extra_module not in sys.modules, f"{extra_module} was imported."


def test_cli_speed_regression_no_torch_dependency():
    # Our CLI should have a relatively clean set of imports.
    # Importing torch is a sign that we are importing too much.
    set_start_method("spawn", force=True)
    process = Process(target=_verify_no_extra_import, args=["torch"])
    process.start()
    process.join()
    assert process.exitcode == 0, "Torch was imported in the CLI. This is a regression."


def test_cli_speed_regression_no_core_dependency():
    # Our CLI should have a relatively clean set of imports.
    # Importing oumi.core is a sign that we are importing too much.
    set_start_method("spawn", force=True)
    process = Process(target=_verify_no_extra_import, args=["oumi.core"])
    process.start()
    process.join()
    assert process.exitcode == 0, (
        "oumi.core was imported in the CLI. This is a regression."
    )
