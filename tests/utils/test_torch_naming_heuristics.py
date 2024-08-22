import torch.nn as nn
import transformers

from lema.utils.torch_naming_heuristics import disable_dropout, group_trainable_params


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
