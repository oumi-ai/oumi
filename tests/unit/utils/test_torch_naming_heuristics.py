import pytest
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
)
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)

from oumi.utils.torch_naming_heuristics import (
    disable_dropout,
    group_trainable_params,
    guess_transformer_layer_cls,
    resolve_transformer_layer_cls_string_as_module_set,
    simplify_transformer_layer_cls_string,
)
from tests.markers import requires_hf_token


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


MODEL_CONFIGS = [
    ("gpt2", "GPT2Block", AutoModelForCausalLM),
    ("facebook/opt-125m", "OPTDecoderLayer", AutoModelForCausalLM),
    ("bert-base-uncased", "BertLayer", AutoModelForCausalLM),
    ("roberta-base", "RobertaLayer", AutoModelForCausalLM),
    ("t5-small", "T5Block", AutoModel),
    (
        "HuggingFaceFW/ablation-model-fineweb-v1",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    (
        "meta-llama/Llama-3.3-70B-Instruct",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    (
        "meta-llama/Llama-3.2-1B-Instruct",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    (
        "meta-llama/Llama-3.2-3B-Instruct",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    (
        "meta-llama/Llama-3.1-8B-Instruct",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    (
        "meta-llama/Llama-3.1-70B-Instruct",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    (
        "meta-llama/Llama-3.1-405B-Instruct",
        "LlamaDecoderLayer",
        AutoModelForCausalLM,
    ),
    ("meta-llama/Meta-Llama-3-8B-Instruct", "LlamaDecoderLayer", AutoModelForCausalLM),
    ("meta-llama/Meta-Llama-3-70B-Instruct", "LlamaDecoderLayer", AutoModelForCausalLM),
    ("microsoft/Phi-3-mini-4k-instruct", "Phi3DecoderLayer", AutoModelForCausalLM),
    # Only available on nightly build
    # ("Qwen/Qwen2-VL-2B-Instruct", "QwenDecoderLayer", AutoModelForVision2Seq),
    ("llava-hf/llava-1.5-7b-hf", "CLIPEncoderLayer", AutoModelForVision2Seq),
    ("Salesforce/blip2-opt-2.7b", "Blip2EncoderLayer", AutoModelForVision2Seq),
    ("mistralai/Mistral-7B-v0.1", "MistralDecoderLayer", AutoModelForCausalLM),
    ("google/gemma-2-2b-it", "GemmaDecoderLayer", AutoModelForCausalLM),
    ("google/gemma-2-2b", "GemmaDecoderLayer", AutoModelForCausalLM),
]


def _load_model_architecture(model_name, builder_class):
    # Loads only the configuration to avoid downloading the full model
    # If possible, load on meta device to avoid initializing weights on CPU
    # Uses CPU for models that don't support `meta`
    try:
        with torch.device("cpu"):
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = builder_class.from_config(config, trust_remote_code=True)
    except RuntimeError as e:
        if "Tensor.item() cannot be called on meta tensors" in str(e):
            model = builder_class.from_config(config, trust_remote_code=True)
        else:
            raise e

    return config, model


@pytest.mark.skip("Very slow test. Only run occasionally if changing that logic.")
@pytest.mark.parametrize(
    "model_name, expected_layer_name, builder_class", MODEL_CONFIGS
)
@requires_hf_token()
def test_guess_transformer_layer_cls(model_name, expected_layer_name, builder_class):
    _config, model = _load_model_architecture(model_name, builder_class)

    # Guess the transformer layer class
    layer_cls = guess_transformer_layer_cls(model)

    # Check if the guessed class name matches the expected name
    assert layer_cls.__name__ == expected_layer_name


@pytest.mark.parametrize(
    "input_name, simplified_name",
    [
        ("", ""),
        ("  \t\n", ""),
        ("Foo", "Foo"),
        (" Foo ", "Foo"),
        (" Foo, Bar ", "Foo,Bar"),
        ("zoo.Foo, Bar ", "Foo,Bar"),
        ("zoo.Foo,moo.Bar ", "Foo,Bar"),
        ("zoo.Foo,,,Zzz,moo.Bar ", "Foo,Zzz,Bar"),
    ],
)
def test_simplify_transformer_layer_cls_string(input_name: str, simplified_name: str):
    assert simplify_transformer_layer_cls_string(input_name) == simplified_name


def test_resolve_transformer_layer_cls_string_as_module_set():
    assert resolve_transformer_layer_cls_string_as_module_set("") == set()

    assert resolve_transformer_layer_cls_string_as_module_set(
        "transformers.models.mllama.modeling_mllama.MllamaCrossAttentionDecoderLayer"
    ) == set(
        {
            MllamaCrossAttentionDecoderLayer,
        }
    )

    assert resolve_transformer_layer_cls_string_as_module_set(
        "transformers.models.mllama.modeling_mllama.MllamaSelfAttentionDecoderLayer,"
        "transformers.models.mllama.modeling_mllama.MllamaCrossAttentionDecoderLayer,"
        "transformers.models.mllama.modeling_mllama.MllamaVisionEncoderLayer"
    ) == set(
        {
            MllamaSelfAttentionDecoderLayer,
            MllamaCrossAttentionDecoderLayer,
            MllamaVisionEncoderLayer,
        }
    )
