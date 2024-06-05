import pytest

from lema import infer, infer_interactive
from lema.core.types import GenerationConfig, InferenceConfig, ModelParams

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "Hello world!\n\nI'm not"


def test_basic_infer_interactive(monkeypatch: pytest.MonkeyPatch):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
        generation=GenerationConfig(
            max_new_tokens=5,
        ),
    )

    # Simulate the user entering "Hello world!" in the terminal:
    monkeypatch.setattr("builtins.input", lambda _: FIXED_PROMPT)
    infer_interactive(config)


def test_basic_infer_non_interactive():
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
        ),
        generation=GenerationConfig(
            max_new_tokens=5,
        ),
    )

    output = infer(
        config,
        [
            [
                FIXED_PROMPT,
            ],
        ],
    )
    assert output == [
        [
            FIXED_RESPONSE,
        ],
    ]
    output = infer(
        config,
        [
            [
                FIXED_PROMPT,
                FIXED_PROMPT,
            ],
            [
                FIXED_PROMPT,
                FIXED_PROMPT,
            ],
        ],
    )
    assert output == [
        [
            FIXED_RESPONSE,
            FIXED_RESPONSE,
        ],
        [
            FIXED_RESPONSE,
            FIXED_RESPONSE,
        ],
    ]
