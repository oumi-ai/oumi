from lema.utils.str_utils import sanitize_run_name


def test_sanitize_run_name_empty():
    assert sanitize_run_name("") == ""


def test_sanitize_run_name_below_max_length_limit():
    assert sanitize_run_name("abc.XYZ-0129_") == "abc.XYZ-0129_"
    assert sanitize_run_name("a_X-7." * 16) == "a_X-7." * 16
    assert sanitize_run_name("a" * 99) == "a" * 99
    assert sanitize_run_name("X" * 100) == "X" * 100


def test_sanitize_run_name_below_invalid_chars():
    assert sanitize_run_name("abc?XYZ/0129^") == "abc_XYZ_0129_"
    assert sanitize_run_name("Лемма") == "_____"


def test_sanitize_run_name_too_long():
    raw_long_run_name = (
        "llama2b.pt.FSDP.HYBRID_SHARD.4node.4xA10040GB.20steps.bs16.gas16.v907."
        "sky-2024-07-22-16-26-33-541717_xrdaukar-4node4gpu-01-lema-cluster"
    )
    actual = sanitize_run_name(raw_long_run_name)
    assert len(actual) == 100
    expected = (
        "llama2b.pt.FSDP.HYBRID_SHARD.4node.4xA10040GB.20steps.bs16.gas16.v907."
        "sky-2024-07...addf327b6c0264e4"
    )
    assert actual == expected
    # verify it's idempotent
    assert sanitize_run_name(actual) == expected
