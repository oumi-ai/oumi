import pytest

from lema import infer, infer_interactive
from lema.core.configs import GenerationConfig, InferenceConfig, ModelParams

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."


def test_infer_basic_interactive(monkeypatch: pytest.MonkeyPatch):
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


@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_infer_basic_non_interactive(num_batches, batch_size):
    model_params = ModelParams(
        model_name="openai-community/gpt2", trust_remote_code=True
    )
    generation_config = GenerationConfig(max_new_tokens=5, batch_size=batch_size)

    input = []
    for _ in range(num_batches):
        for _ in range(batch_size):
            input.append(FIXED_PROMPT)
    output = infer(
        model_params=model_params,
        generation_config=generation_config,
        input=input,
    )

    expected_output = []
    for _ in range(num_batches):
        for _ in range(batch_size):
            expected_output.append(FIXED_RESPONSE)
    assert output == expected_output
