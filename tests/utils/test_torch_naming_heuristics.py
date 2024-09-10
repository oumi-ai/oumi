import pytest
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModel

from lema.utils.torch_naming_heuristics import (
    disable_dropout,
    group_trainable_params,
    guess_transformer_layer_cls,
)


def test_disable_dropout():
    config = transformers.GPT2Config()
    config.attn_pdrop = 0.1
    config.embd_pdrop = 0.2
    config.resid_pdrop = 0.3
    config.summary_first_dropout = 0.4
    config.vocab_size = 5
    config.initializer_range = 0.06

    disable_dropout(config)

    assert config.attn_pdrop == 0.0
    assert config.embd_pdrop == 0.0
    assert config.resid_pdrop == 0.0
    assert config.summary_first_dropout == 0.0
    assert config.vocab_size == 5
    assert config.initializer_range == 0.06


def test_group_trainable_params():
    embedding = nn.Embedding(20, 10)
    linear_bias = nn.Linear(10, 10, bias=True)
    layernorm = nn.LayerNorm(10)
    model = nn.ModuleList([embedding, linear_bias, layernorm])

    decay_params = [embedding.weight, linear_bias.weight]
    nodecay_params = [linear_bias.bias, layernorm.weight, layernorm.bias]
    expected = [
        {
            "params": decay_params,
            "weight_decay": 0.1,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]

    assert group_trainable_params(model, 0.1) == expected


def test_guess_transformer_layer_empty_model():
    # Test with an empty model
    empty_model = nn.Module()
    with pytest.raises(ValueError, match="Unable to guess transformer layer class"):
        guess_transformer_layer_cls(empty_model)


@pytest.mark.parametrize(
    "model_name, expected_layer_name",
    [
        ("gpt2", "GPT2Block"),
        ("facebook/opt-125m", "OPTDecoderLayer"),
        ("meta-llama/Llama-2-7b-hf", "LlamaDecoderLayer"),
        ("bert-base-uncased", "BertLayer"),
        ("roberta-base", "RobertaLayer"),
        ("t5-small", "T5Block"),
    ],
)
def test_guess_transformer_layer_cls_huggingface_models(
    model_name, expected_layer_name
):
    # Load only the configuration to avoid downloading the full model
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_config(config)

    # Guess the transformer layer class
    layer_cls = guess_transformer_layer_cls(model)

    assert (
        layer_cls.__name__ == expected_layer_name
    ), f"Expected {expected_layer_name}, but got {layer_cls.__name__}"

    # Additional check: ensure the guessed class is actually used in the model
    found = False
    for module in model.modules():
        if isinstance(module, layer_cls):
            found = True
            break
    assert found, f"Guessed layer class {layer_cls.__name__} not found in model"
