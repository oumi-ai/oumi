import pytest

from lema import infer_prob
from lema.core.types import ModelParams

PROMPTS = [
    [
        "The first letter of the english alphabet is ",
        "United States is in North ",
    ],
    [
        "The name 'Carole' starts with ",
        "Copenhagen is the capital of ",
    ],
]
POSSIBLE_ANSWERS = ["A", "B", "C", "D"]
CORRECT_ANSWERS = [
    [
        ["A"],
        ["A"],
    ],
    [
        ["C"],
        ["D"],
    ],
]


@pytest.mark.parametrize("num_batches,batch_size", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_infer_prob(num_batches, batch_size):
    model_params = ModelParams(
        model_name="openai-community/gpt2", trust_remote_code=True
    )

    input = []
    for batch_no in range(num_batches):
        batch_input = []
        for batch_index in range(batch_size):
            batch_input.append(PROMPTS[batch_no][batch_index])
        input.append(batch_input)

    output = infer_prob(
        model_params=model_params,
        input=input,
        acceptable_logits=POSSIBLE_ANSWERS,
    )

    for batch_no, batch in enumerate(output):
        for batch_index, probs in enumerate(batch):
            answer_index = probs.index(max(probs))
            answer = POSSIBLE_ANSWERS[answer_index]
            assert answer == CORRECT_ANSWERS[batch_no][batch_index][0]
