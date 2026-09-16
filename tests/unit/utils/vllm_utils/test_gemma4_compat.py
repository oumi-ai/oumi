import importlib.metadata
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from oumi.utils.vllm_utils.gemma4_compat import (
    ACTIVATION_ENV_VAR,
    register_gemma4_compatibility,
)


class _FakeLinear:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, hidden_states):
        return hidden_states, None


class _FakeNorm:
    def __call__(self, value):
        return value


class _FakeRotaryEmbedding:
    def __init__(self):
        self.last_key = "not-called"

    def __call__(self, positions, query, key):
        self.last_key = key
        return query, key


class _FakeAttentionOperation:
    def __init__(self):
        self.last_key = "not-called"
        self.last_value = "not-called"

    def __call__(self, query, key, value):
        self.last_key = key
        self.last_value = value
        return query


class _FakeGemma4Attention:
    original_forward_calls = 0

    def __init__(
        self,
        config,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim,
        max_position_embeddings,
        use_k_eq_v=False,
        cache_config=None,
        quant_config=None,
        attn_logits_soft_cap=None,
        prefix="",
    ):
        del max_position_embeddings, use_k_eq_v, cache_config
        del quant_config, attn_logits_soft_cap
        self.config = config
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_kv_shared_layer = int(prefix.split(".layers.")[1].split(".")[0]) >= (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        self.qkv_proj = _FakeLinear()
        self.q_norm = _FakeNorm()
        self.k_norm = _FakeNorm()
        self.v_norm = _FakeNorm()
        self.rotary_emb = _FakeRotaryEmbedding()
        self.attn = _FakeAttentionOperation()
        self.o_proj = _FakeLinear()

    def forward(self, positions, hidden_states, **kwargs):
        del positions, hidden_states, kwargs
        type(self).original_forward_calls += 1
        return "original-forward"


class _FakeGemma4ForCausalLM:
    def __init__(self, required_weights=()):
        self.config = SimpleNamespace(
            num_hidden_layers=6,
            num_kv_shared_layers=2,
        )
        self.required_weights = set(required_weights)
        self.received_weights = []

    def load_weights(self, weights):
        self.received_weights = [name for name, _ in weights]
        missing_weights = self.required_weights - set(self.received_weights)
        if missing_weights:
            raise ValueError(f"Missing weights: {sorted(missing_weights)}")
        return set(self.received_weights)


@pytest.fixture
def fake_gemma4(monkeypatch):
    class FakeGemma4Attention(_FakeGemma4Attention):
        pass

    class FakeGemma4ForCausalLM(_FakeGemma4ForCausalLM):
        pass

    gemma4_module = ModuleType("vllm.model_executor.models.gemma4")
    setattr(gemma4_module, "Gemma4Attention", FakeGemma4Attention)
    setattr(gemma4_module, "Gemma4ForCausalLM", FakeGemma4ForCausalLM)
    setattr(gemma4_module, "ColumnParallelLinear", _FakeLinear)

    vllm_module = ModuleType("vllm")
    model_executor_module = ModuleType("vllm.model_executor")
    models_module = ModuleType("vllm.model_executor.models")
    setattr(models_module, "gemma4", gemma4_module)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", model_executor_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.gemma4", gemma4_module)
    return gemma4_module


def _enable_plugin(monkeypatch, version="0.19.1"):
    monkeypatch.setenv(ACTIVATION_ENV_VAR, "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda package: version)


def _build_attention(gemma4_module, layer_index):
    config = SimpleNamespace(
        attention_bias=False,
        num_hidden_layers=6,
        num_kv_shared_layers=2,
    )
    return gemma4_module.Gemma4Attention(
        config=config,
        hidden_size=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        max_position_embeddings=128,
        prefix=f"model.layers.{layer_index}.self_attn",
    )


def test_registration_requires_activation(monkeypatch, fake_gemma4):
    monkeypatch.delenv(ACTIVATION_ENV_VAR, raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.19.1")
    original_init = fake_gemma4.Gemma4Attention.__init__

    register_gemma4_compatibility()

    assert fake_gemma4.Gemma4Attention.__init__ is original_init


def test_registration_requires_exact_vllm_version(monkeypatch, fake_gemma4):
    _enable_plugin(monkeypatch, version="0.19.2")
    original_init = fake_gemma4.Gemma4Attention.__init__

    register_gemma4_compatibility()

    assert fake_gemma4.Gemma4Attention.__init__ is original_init


def test_registration_is_idempotent_and_logs_process_marker(
    monkeypatch, fake_gemma4, caplog
):
    _enable_plugin(monkeypatch)
    caplog.set_level(logging.INFO)

    register_gemma4_compatibility()
    installed_init = fake_gemma4.Gemma4Attention.__init__
    register_gemma4_compatibility()

    assert fake_gemma4.Gemma4Attention.__init__ is installed_init
    installation_messages = [
        record.message
        for record in caplog.records
        if "[oumi-gemma4-compat] installed" in record.message
    ]
    assert len(installation_messages) == 1
    assert "pid=" in installation_messages[0]
    assert "process=" in installation_messages[0]


def test_shared_attention_uses_q_only_topology(monkeypatch, fake_gemma4):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()

    attention = _build_attention(fake_gemma4, layer_index=4)

    assert hasattr(attention, "q_proj")
    assert not hasattr(attention, "qkv_proj")
    assert not hasattr(attention, "k_norm")
    assert not hasattr(attention, "v_norm")
    assert attention.q_proj.args == (4, 4)
    assert attention.q_proj.kwargs["prefix"] == "model.layers.4.self_attn.q_proj"


def test_non_shared_attention_keeps_original_topology(monkeypatch, fake_gemma4):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()

    attention = _build_attention(fake_gemma4, layer_index=3)

    assert hasattr(attention, "qkv_proj")
    assert hasattr(attention, "k_norm")
    assert hasattr(attention, "v_norm")
    assert not hasattr(attention, "q_proj")


def test_shared_forward_passes_none_for_key_and_value(monkeypatch, fake_gemma4):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()
    attention = _build_attention(fake_gemma4, layer_index=4)
    hidden_states = torch.arange(4).reshape(1, 4)

    output = attention.forward(torch.tensor([0]), hidden_states)

    assert torch.equal(output, hidden_states)
    assert attention.rotary_emb.last_key is None
    assert attention.attn.last_key is None
    assert attention.attn.last_value is None


def test_non_shared_forward_remains_unchanged(monkeypatch, fake_gemma4):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()
    attention = _build_attention(fake_gemma4, layer_index=3)
    original_call_count = fake_gemma4.Gemma4Attention.original_forward_calls

    output = attention.forward(torch.tensor([0]), torch.zeros(1, 4))

    assert output == "original-forward"
    assert fake_gemma4.Gemma4Attention.original_forward_calls == original_call_count + 1


def test_minimal_checkpoint_loads_without_shared_kv_weights(monkeypatch, fake_gemma4):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()
    required_weights = {
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.o_proj.weight",
        "model.layers.4.mlp.down_proj.weight",
    }
    model = fake_gemma4.Gemma4ForCausalLM(required_weights)
    weights = [(name, torch.empty(0)) for name in required_weights]

    loaded = model.load_weights(weights)

    assert loaded == required_weights


def test_legacy_checkpoint_filters_only_redundant_shared_weights(
    monkeypatch, fake_gemma4
):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()
    expected_weights = {
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.o_proj.weight",
    }
    redundant_weights = {
        "model.layers.4.self_attn.k_proj.weight",
        "model.layers.4.self_attn.v_proj.weight",
        "model.layers.4.self_attn.k_norm.weight",
        "model.layers.5.self_attn.k_proj.weight",
    }
    model = fake_gemma4.Gemma4ForCausalLM(expected_weights)
    weights = [(name, torch.empty(0)) for name in expected_weights | redundant_weights]

    loaded = model.load_weights(weights)

    assert loaded == expected_weights
    assert set(model.received_weights) == expected_weights


@pytest.mark.parametrize(
    "missing_weight",
    [
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.o_proj.weight",
        "model.layers.4.mlp.down_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
    ],
)
def test_genuine_missing_weights_remain_errors(
    monkeypatch, fake_gemma4, missing_weight
):
    _enable_plugin(monkeypatch)
    register_gemma4_compatibility()
    required_weights = {
        "model.layers.4.self_attn.q_proj.weight",
        "model.layers.4.self_attn.o_proj.weight",
        "model.layers.4.mlp.down_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
    }
    model = fake_gemma4.Gemma4ForCausalLM(required_weights)
    weights = [
        (name, torch.empty(0)) for name in required_weights if name != missing_weight
    ]

    with pytest.raises(ValueError, match="Missing weights"):
        model.load_weights(weights)


def test_distribution_registers_vllm_general_plugin():
    entry_points = importlib.metadata.entry_points(group="vllm.general_plugins")
    matching_entry_points = [
        entry_point
        for entry_point in entry_points
        if entry_point.name == "oumi_gemma4_compat"
    ]

    assert len(matching_entry_points) == 1
    assert (
        matching_entry_points[0].value
        == "oumi.utils.vllm_utils.gemma4_compat:register_gemma4_compatibility"
    )
