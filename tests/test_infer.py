import pytest

from lema import infer
from lema.core.types import GenerationConfig, InferenceConfig, ModelParams


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
    monkeypatch.setattr("builtins.input", lambda _: "Hello world!")
    infer(config, interactive=True)

    # TODO: Change "infer" interface to return output for testing.


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

    with pytest.raises(NotImplementedError) as exception_info:
        infer(config, interactive=False)

    assert str(exception_info.value) == (
        "Non-interactive inference is not implemented yet"
    )
