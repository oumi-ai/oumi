import pytest

from lema.core.types import ModelParams
from lema.core.types.base_tokenizer import BaseTokenizer
from lema.evaluation import infer_prob, most_probable_tokens

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
        acceptable_tokens=POSSIBLE_ANSWERS,
    )

    for batch_no, batch in enumerate(output):
        for batch_index, probs in enumerate(batch):
            answer_index = probs.index(max(probs))
            answer = POSSIBLE_ANSWERS[answer_index]
            assert answer == CORRECT_ANSWERS[batch_no][batch_index][0]


def test_infer_prob_entire_vocab():
    model_params = ModelParams(
        model_name="openai-community/gpt2", trust_remote_code=True
    )

    input = [
        [
            "Some random text ",
        ],
    ]
    output = infer_prob(model_params=model_params, input=input)
    assert len(output[0][0]) == 50257  # 50257 is the GPT-2 vocab size


@pytest.mark.parametrize(
    "token_probs,output",
    [
        [[0.99, 0.01], [("index_0", 0.99), ("index_1", 0.01)]],
        [[0.01, 0.99], [("index_1", 0.99), ("index_0", 0.01)]],
        [[0.5, 0.5], [("index_1", 0.5), ("index_0", 0.5)]],
        [
            [0.25, 0.4, 0.25, 0.1],
            [("index_1", 0.4), ("index_2", 0.25), ("index_0", 0.25)],
        ],
        [
            [0.01, 0.01, 0.01, 0.01, 0.01, 0.8, 0.01, 0.01, 0.03, 0.1],
            [("index_5", 0.8), ("index_9", 0.1), ("index_8", 0.03)],
        ],
    ],
)
def test_most_probable_tokens(token_probs, output):
    class MockDecoder(BaseTokenizer):
        def decode(self, x):
            return f"index_{x}"

    assert most_probable_tokens(MockDecoder(), token_probs, count=3) == output


@pytest.mark.parametrize("count", [0, 1, 2, 3, 1000])
def test_most_probable_tokens_varying_count(count):
    class MockDecoder(BaseTokenizer):
        def decode(self, x):
            return f"index_{x}"

    token_probs = [0.8, 0.7, 0.3, 0.2, 0.0, 0.1, 0.4, 0.5, 0.9, 0.6]
    full_output = [
        ("index_8", 0.9),
        ("index_0", 0.8),
        ("index_1", 0.7),
        ("index_9", 0.6),
        ("index_7", 0.5),
        ("index_6", 0.4),
        ("index_2", 0.3),
        ("index_3", 0.2),
        ("index_5", 0.1),
        ("index_4", 0.0),
    ]
    assert (
        most_probable_tokens(MockDecoder(), token_probs, count) == full_output[:count]
    )
