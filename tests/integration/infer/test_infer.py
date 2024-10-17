import pytest

from oumi import infer, infer_interactive
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."


def test_infer_basic_interactive(monkeypatch: pytest.MonkeyPatch):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="openai-community/gpt2",
            trust_remote_code=True,
            chat_template="gpt2",
            tokenizer_pad_token="<|endoftext|>",
        ),
        generation=GenerationParams(max_new_tokens=5, temperature=0.0, seed=42),
    )

    # Simulate the user entering "Hello world!" in the terminal folowed by Ctrl+D.
    input_iterator = iter([FIXED_PROMPT])

    def mock_input(_):
        try:
            return next(input_iterator)
        except StopIteration:
            raise EOFError  # Simulate Ctrl+D

    # Replace the built-in input function
    monkeypatch.setattr("builtins.input", mock_input)
    infer_interactive(config)


@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_infer_basic_non_interactive(num_batches, batch_size):
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
    )
    generation_params = GenerationParams(
        max_new_tokens=5, temperature=0.0, seed=42, batch_size=batch_size
    )

    input = [FIXED_PROMPT] * (num_batches * batch_size)
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
    )

    conversation = Conversation(
        messages=(
            [
                Message(content=FIXED_PROMPT, role=Role.USER),
                Message(content=FIXED_RESPONSE, role=Role.ASSISTANT),
            ]
        )
    )
    expected_output = [conversation] * (num_batches * batch_size)
    assert output == expected_output
