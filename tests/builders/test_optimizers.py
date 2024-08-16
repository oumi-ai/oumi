import torch.nn as nn

from lema.builders.optimizers import _group_trainable_params


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

    assert _group_trainable_params(model, 0.1) == expected
