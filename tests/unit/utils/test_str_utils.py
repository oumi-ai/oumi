import os
import re
from typing import Optional
from unittest.mock import patch

import pytest

from oumi.builders import build_tokenizer
from oumi.core.configs import ModelParams
from oumi.utils.str_utils import (
    compute_utf8_len,
    get_editable_install_override_env_var,
    sanitize_run_name,
    set_oumi_install_editable,
    str_to_bool,
    truncate_text_pieces_to_max_tokens_limit,
    truncate_to_max_tokens_limit,
    try_str_to_bool,
)


@pytest.fixture
def gpt2_tokenizer():
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="float16",
            trust_remote_code=False,
            chat_template="default",
            tokenizer_pad_token="<|endoftext|>",
        )
    )
    assert tokenizer.pad_token_id is not None
    assert isinstance(tokenizer.pad_token_id, int)
    return tokenizer


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
        "fineweb.pt.FSDP.HYBRID_SHARD.4node.4xA10040GB.20steps.bs16.gas16.v907."
        "sky-2024-07-22-16-26-33-541717_xrdaukar-4node4gpu-01-oumi-cluster"
    )
    actual = sanitize_run_name(raw_long_run_name)
    assert actual is not None
    assert len(actual) == 100
    expected = (
        "fineweb.pt.FSDP.HYBRID_SHARD.4node.4xA10040GB.20steps.bs16.gas16.v907."
        "sky-2024-07...9db6edce8186fb6a"
    )
    assert actual == expected
    # verify it's idempotent
    assert sanitize_run_name(actual) == expected


@pytest.mark.parametrize(
    "value",
    ["true", "True", "TRUE", "yes", "Yes", "YES", "1", "on", "ON", "t", "y", " True "],
)
def test_true_values(value):
    assert str_to_bool(value) is True
    assert try_str_to_bool(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "false",
        "False",
        "FALSE",
        "no",
        "No",
        "NO",
        "0",
        "off",
        "OFF",
        "f",
        "n",
        " False ",
    ],
)
def test_false_values(value):
    assert str_to_bool(value) is False
    assert try_str_to_bool(value) is False


@pytest.mark.parametrize("value", ["maybe", "unknown", "tru", "ye", "2", "nope"])
def test_invalid_inputs(value):
    assert try_str_to_bool(value) is None
    with pytest.raises(ValueError):
        str_to_bool(value)


def test_compute_utf8_len():
    assert compute_utf8_len("") == 0
    assert compute_utf8_len("a") == 1
    assert compute_utf8_len("abc") == 3
    assert compute_utf8_len("a b c") == 5
    assert compute_utf8_len("Wir müssen") == 11
    assert compute_utf8_len("Мы должны") == 17


@pytest.mark.parametrize(
    "env_var_val,expected_val",
    [
        (None, False),
        ("1", True),
        ("true", True),
        ("True", True),
        ("0", False),
        ("false", False),
        ("False", False),
    ],
)
def test_get_editable_install_override(env_var_val: Optional[str], expected_val: bool):
    overrides = {}
    if env_var_val is not None:
        overrides = {"OUMI_FORCE_EDITABLE_INSTALL": env_var_val}
    with patch.dict(os.environ, overrides, clear=True):
        assert get_editable_install_override_env_var() == expected_val


@pytest.mark.parametrize(
    "setup,output_setup",
    [
        (
            "pip install 'oumi[gpu]'",
            "pip install -e '.[gpu]'",
        ),
        (
            """
            #A comment
            pip install -e uv && uv pip -q install "oumi[gpu,dev]" vllm # comment
            pip install -e "oumi"
            """,
            """
            #A comment
            pip install -e uv && uv pip -q install -e '.[gpu,dev]' vllm # comment
            pip install -e "oumi"
            """,
        ),
        (
            """
            #A comment
            pip -q --debug install -U "skypilot[azure]" oumi vllm "wandb" # Foo.
            print("All done")
            """,
            """
            #A comment
            pip -q --debug install -U "skypilot[azure]" -e '.' vllm "wandb" # Foo.
            print("All done")
            """,
        ),
    ],
)
def test_set_oumi_install_editable(setup, output_setup):
    actual_setup = set_oumi_install_editable(setup)
    assert actual_setup == output_setup


@pytest.mark.parametrize(
    "text,max_tokens,truncation_side,expected_text,expected_tokens",
    [
        ("Hello World!", 100, "right", "Hello World!", 3),
        ("Hello World!", 100, "left", "Hello World!", 3),
        ("Hello World!", 3, "right", "Hello World!", 3),
        ("Hello World!", 3, "left", "Hello World!", 3),
        ("Hello World!", 2, "right", "Hello World", 2),
        ("Hello World!", 2, "left", " World!", 2),
        ("Hello World!", 1, "right", "Hello", 1),
        ("Hello World!", 1, "left", "!", 1),
        ("", 10, "right", "", 0),
        ("", 10, "left", "", 0),
        ("", 1, "right", "", 0),
        ("", 1, "left", "", 0),
        ("  \t", 10, "right", "  \t", 3),
        ("  \t\n", 10, "left", "  \t\n", 4),
        ("  \t", 1, "right", " ", 1),
        ("  \t\n", 1, "left", "\n", 1),
        ("Hellolololworld!", 1, "right", "Hell", 1),
        ("Hellolololworld!", 1, "left", "!", 1),
        ("Hellolololworld!", 2, "right", "Hellol", 2),
        ("Hellolololworld!", 2, "left", "world!", 2),
    ],
)
def test_truncate_to_max_tokens_limit_success(
    text: str,
    max_tokens: int,
    truncation_side: str,
    expected_text: str,
    expected_tokens: int,
    gpt2_tokenizer,
):
    truncated_text, truncated_tokens = truncate_to_max_tokens_limit(
        text,
        tokenizer=gpt2_tokenizer,
        max_tokens=max_tokens,
        truncation_side=truncation_side,
    )
    assert truncated_text == expected_text
    assert truncated_tokens == expected_tokens


def test_truncate_to_max_tokens_limit_invalid_args(
    gpt2_tokenizer,
):
    with pytest.raises(
        ValueError, match=re.escape("`max_tokens` must be a positive integer")
    ):
        truncated_text, truncated_tokens = truncate_to_max_tokens_limit(
            "hi there",
            tokenizer=gpt2_tokenizer,
            max_tokens=0,
            truncation_side="right",
        )

    with pytest.raises(
        ValueError, match=re.escape("`max_tokens` must be a positive integer")
    ):
        truncated_text, truncated_tokens = truncate_to_max_tokens_limit(
            "apple",
            tokenizer=gpt2_tokenizer,
            max_tokens=-1,
            truncation_side="left",
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid truncation_side: 'kaboom'. Expected 'left' or 'right'"
        ),
    ):
        truncated_text, truncated_tokens = truncate_to_max_tokens_limit(
            "hi there",
            tokenizer=gpt2_tokenizer,
            max_tokens=100,
            truncation_side="kaboom",
        )


@pytest.mark.parametrize(
    "text_pieces,max_tokens,truncation_side,expected_text_pieces",
    [
        (["Hello", "World!"], 100, "right", None),
        (["Hello", "World!"], 100, "left", None),
        (["Hello", "World!"], 2, "right", ["Hello", "World"]),
        (["Hello", "World!"], 2, "left", ["", "World!"]),
        (["Hello", "World!"], 1, "right", ["Hello", ""]),
        (["Hello", "World!"], 1, "left", ["", "!"]),
    ],
)
def test_truncate_text_pieces_to_max_tokens_limit_success(
    text_pieces: list[str],
    max_tokens: int,
    truncation_side: str,
    expected_text_pieces: Optional[list[str]],
    gpt2_tokenizer,
):
    truncated_text_pieces = truncate_text_pieces_to_max_tokens_limit(
        text_pieces,
        tokenizer=gpt2_tokenizer,
        max_tokens=max_tokens,
        truncation_side=truncation_side,
    )

    if expected_text_pieces is not None:
        assert truncated_text_pieces == expected_text_pieces
    else:
        assert truncated_text_pieces == text_pieces
