from unittest import mock
from unittest.mock import patch

import pytest

from oumi.core.configs import GenerationParams, ModelParams
from oumi.inference import NativeTextInferenceEngine

_TEXT_ONLY = "oumi.inference.native_text_inference_engine"


def _mock_tokenizer(padding_side: str = "right") -> mock.MagicMock:
    tokenizer = mock.MagicMock()
    tokenizer.padding_side = padding_side
    tokenizer.pad_token_id = 0
    tokenizer.eos_token = "<eos>"
    tokenizer.eos_token_id = 2
    return tokenizer


def _build_engine(
    tokenizer: mock.MagicMock,
    *,
    is_vlm: bool = False,
    tokenizer_kwargs: dict | None = None,
) -> NativeTextInferenceEngine:
    model = mock.MagicMock()
    processor = mock.MagicMock() if is_vlm else None
    with patch.multiple(
        _TEXT_ONLY,
        build_model=mock.MagicMock(return_value=model),
        build_tokenizer=mock.MagicMock(return_value=tokenizer),
        build_processor=mock.MagicMock(return_value=processor),
        is_image_text_llm=mock.MagicMock(return_value=is_vlm),
        find_internal_model_config_using_model_name=mock.MagicMock(return_value=None),
    ):
        return NativeTextInferenceEngine(
            ModelParams(
                model_name="test-model", tokenizer_kwargs=tokenizer_kwargs or {}
            ),
            generation_params=GenerationParams(),
        )


def test_text_engine_left_pads_for_batched_generation():
    tokenizer = _mock_tokenizer("right")

    _build_engine(tokenizer)

    assert tokenizer.padding_side == "left"


def test_text_engine_respects_explicit_padding_side():
    tokenizer = _mock_tokenizer("right")

    _build_engine(tokenizer, tokenizer_kwargs={"padding_side": "right"})

    assert tokenizer.padding_side == "right"


def test_vlm_engine_keeps_tokenizer_padding_side():
    tokenizer = _mock_tokenizer("right")

    _build_engine(tokenizer, is_vlm=True)

    assert tokenizer.padding_side == "right"


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_text_engine_leaves_left_padding_untouched(padding_side: str):
    tokenizer = _mock_tokenizer(padding_side)

    _build_engine(tokenizer)

    assert tokenizer.padding_side == "left"
