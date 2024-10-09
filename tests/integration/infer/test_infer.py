import pytest

from oumi import infer, infer_interactive
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."


def test_infer_basic_interactive(monkeypatch: pytest.MonkeyPatch):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
            chat_template="gpt2",
        ),
        generation=GenerationParams(max_new_tokens=5, temperature=0.0, seed=42),
    )

    # Simulate the user entering "Hello world!" in the terminal:
    monkeypatch.setattr("builtins.input", lambda _: FIXED_PROMPT)
    infer_interactive(config)


@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_infer_basic_non_interactive(num_batches, batch_size):
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
    )
    generation_params = GenerationParams(
        max_new_tokens=5, temperature=0.0, seed=42, batch_size=batch_size
    )

    input = []
    for _ in range(num_batches):
        for _ in range(batch_size):
            input.append(FIXED_PROMPT)
    output = infer(
        model_params=model_params,
        generation_params=generation_params,
        input=input,
    )

    expected_output = []
    for _ in range(num_batches):
        for _ in range(batch_size):
            expected_output.append(FIXED_RESPONSE)
    assert output == expected_output
