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


MODEL_CONFIGS = [
    ("gpt2", "GPT2Block"),
    ("facebook/opt-125m", "OPTDecoderLayer"),
    ("meta-llama/Llama-2-7b-hf", "LlamaDecoderLayer"),
    ("bert-base-uncased", "BertLayer"),
    ("roberta-base", "RobertaLayer"),
    ("t5-small", "T5Block"),
    ("HuggingFaceFW/ablation-model-fineweb-v1", "LlamaDecoderLayer"),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "LlamaDecoderLayer"),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct", "LlamaDecoderLayer"),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "LlamaDecoderLayer"),
    ("meta-llama/Meta-Llama-3-70B-Instruct", "LlamaDecoderLayer"),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi3DecoderLayer"),
    ("Qwen/Qwen2-VL-2B-Instruct", "QwenDecoderLayer"),
    ("llava-hf/llava-1.5-7b-hf", "LlavaDecoderLayer"),
    ("Salesforce/blip2-opt-2.7b", "Blip2DecoderLayer"),
    ("mistralai/Mistral-7B-v0.1", "MistralDecoderLayer"),
]


def test_guess_transformer_layer_empty_model():
    # Test with an empty model
    empty_model = nn.Module()
    with pytest.raises(ValueError, match="Unable to guess transformer layer class"):
        guess_transformer_layer_cls(empty_model)


@pytest.mark.parametrize("model_name, expected_layer_name", MODEL_CONFIGS)
def test_guess_transformer_layer_cls(model_name, expected_layer_name):
    # Load only the configuration to avoid downloading the full model
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_config(config)

    # Guess the transformer layer class
    layer_cls = guess_transformer_layer_cls(model)

    # Check if the guessed class name matches the expected name
    assert (
        layer_cls.__name__ == expected_layer_name
    ), f"For {model_name}: Expected {expected_layer_name}, but got {layer_cls.__name__}"

    # Verify that the guessed class is actually used in the model
    found = any(isinstance(module, layer_cls) for module in model.modules())
    assert found, f"Guessed layer class {layer_cls.__name__} not found in {model_name}"
