import importlib.metadata
import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from oumi.utils.vllm_utils.gemma4_compat import (
    ACTIVATION_ENV_VAR,
    register_gemma4_compatibility,
)

# The fake config has 6 layers with the last 2 sharing KV, so layer 4 is shared
# and layer 3 is the last ordinary one.
_ORDINARY_WEIGHT = "model.layers.3.self_attn.k_proj.weight"
_SHARED_WEIGHT = "model.layers.4.self_attn.q_proj.weight"


def _identity(value):
    return value


class _FakeLinear:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, hidden_states):
        return hidden_states, None


class _CallRecorder:
    """Records the positional arguments of the last call."""

    def __init__(self, result):
        self.args = ()
        # vLLM's Attention carries this; the patch refuses to build a shared
        # layer when it is set.
        self.calculate_kv_scales = False
        self._result = result

    def __call__(self, *args):
        self.args = args
        return self._result(*args)


class _FakeGemma4Attention:
    """The parts of vLLM 0.19.1's Gemma4Attention that the patch touches."""

    module: ModuleType

    def __init__(
        self,
        config,
        hidden_size,
        num_heads,
        _num_kv_heads,
        head_dim,
        _max_position_embeddings=None,
        _use_k_eq_v=False,
        _cache_config=None,
        _quant_config=None,
        _attn_logits_soft_cap=None,
        prefix="",
    ):
        self.total_num_heads = num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        # vLLM derives this itself rather than calling extract_layer_index here.
        self.is_kv_shared_layer = int(prefix.split(".layers.")[1].split(".")[0]) >= (
            config.num_hidden_layers - config.num_kv_shared_layers
        )
        # Resolved through the module, the way vLLM resolves its own globals.
        self.qkv_proj = self.module.QKVParallelLinear(hidden_size, head_dim)
        self.q_norm = _identity
        self.k_norm = _identity
        self.v_norm = _identity
        self.rotary_emb = _CallRecorder(lambda positions, query, key: (query, key))
        self.attn = _CallRecorder(lambda query, key, value: query)
        self.attn.calculate_kv_scales = self.module.calculate_kv_scales
        self.o_proj = _FakeLinear()

    def forward(self, positions, hidden_states, **kwargs):
        del positions, hidden_states, kwargs
        return "original-forward"


class _FakeGemma4ForCausalLM:
    def __init__(self, required_weights=()):
        self.config = SimpleNamespace(num_hidden_layers=6, num_kv_shared_layers=2)
        self.required_weights = set(required_weights)

    def load_weights(self, weights):
        received_weights = {name for name, _ in weights}
        missing_weights = self.required_weights - received_weights
        if missing_weights:
            raise ValueError(f"Missing weights: {sorted(missing_weights)}")
        return received_weights


@pytest.fixture
def fake_gemma4(monkeypatch):
    gemma4: Any = ModuleType("vllm.model_executor.models.gemma4")
    gemma4.Gemma4Attention = type("Gemma4Attention", (_FakeGemma4Attention,), {})
    gemma4.Gemma4ForCausalLM = type("Gemma4ForCausalLM", (_FakeGemma4ForCausalLM,), {})
    gemma4.ColumnParallelLinear = _FakeLinear
    gemma4.QKVParallelLinear = _FakeLinear
    gemma4.extract_layer_index = lambda prefix: int(
        prefix.split(".layers.")[1].split(".")[0]
    )
    gemma4.calculate_kv_scales = False
    gemma4.Gemma4Attention.module = gemma4

    models: Any = ModuleType("vllm.model_executor.models")
    models.gemma4 = gemma4
    for name, module in {
        "vllm": ModuleType("vllm"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": models,
        "vllm.model_executor.models.gemma4": gemma4,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return gemma4


@pytest.fixture
def installed(monkeypatch, fake_gemma4):
    """The fake gemma4 module with the compatibility patch applied."""
    monkeypatch.setenv(ACTIVATION_ENV_VAR, "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.19.1")
    register_gemma4_compatibility()
    return fake_gemma4


def _build_attention(gemma4, layer_index):
    config = SimpleNamespace(
        attention_bias=False, num_hidden_layers=6, num_kv_shared_layers=2
    )
    return gemma4.Gemma4Attention(
        config=config,
        hidden_size=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        max_position_embeddings=128,
        prefix=f"model.layers.{layer_index}.self_attn",
    )


def _load_weights(gemma4, present, required):
    model = gemma4.Gemma4ForCausalLM(required)
    return model.load_weights([(name, torch.empty(0)) for name in present])


@pytest.mark.parametrize(
    ("activated", "version"),
    [(False, "0.19.1"), (True, "0.19.2")],
    ids=["not-activated", "wrong-vllm-version"],
)
def test_registration_guards(monkeypatch, fake_gemma4, activated, version):
    if activated:
        monkeypatch.setenv(ACTIVATION_ENV_VAR, "1")
    else:
        monkeypatch.delenv(ACTIVATION_ENV_VAR, raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: version)
    original_init = fake_gemma4.Gemma4Attention.__init__

    register_gemma4_compatibility()

    assert fake_gemma4.Gemma4Attention.__init__ is original_init


def test_registration_is_idempotent(monkeypatch, fake_gemma4, caplog):
    monkeypatch.setenv(ACTIVATION_ENV_VAR, "1")
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.19.1")
    caplog.set_level(logging.INFO)

    register_gemma4_compatibility()
    installed_init = fake_gemma4.Gemma4Attention.__init__
    register_gemma4_compatibility()

    assert fake_gemma4.Gemma4Attention.__init__ is installed_init
    assert sum("compat] installed" in r.message for r in caplog.records) == 1


def test_shared_attention_uses_q_only_topology(installed):
    attention = _build_attention(installed, layer_index=4)

    assert not hasattr(attention, "qkv_proj")
    assert not hasattr(attention, "k_norm")
    assert not hasattr(attention, "v_norm")
    assert attention.q_proj.args == (4, 4)
    assert attention.q_proj.kwargs["prefix"] == "model.layers.4.self_attn.q_proj"


def test_ordinary_attention_is_untouched(installed):
    attention = _build_attention(installed, layer_index=3)

    assert hasattr(attention, "qkv_proj")
    assert hasattr(attention, "k_norm")
    assert hasattr(attention, "v_norm")
    assert not hasattr(attention, "q_proj")
    assert attention.forward(torch.tensor([0]), torch.zeros(1, 4)) == "original-forward"


def test_shared_layer_never_builds_the_packed_projection(installed):
    constructed = []

    def _tracking_qkv_parallel_linear(*args, **kwargs):
        constructed.append(args)
        return _FakeLinear(*args, **kwargs)

    installed.QKVParallelLinear = _tracking_qkv_parallel_linear

    _build_attention(installed, layer_index=4)
    assert constructed == []

    _build_attention(installed, layer_index=3)
    assert len(constructed) == 1


def test_kv_scale_calculation_is_rejected_for_shared_layers(installed):
    installed.calculate_kv_scales = True

    _build_attention(installed, layer_index=3)

    with pytest.raises(NotImplementedError, match="calculate-kv-scales"):
        _build_attention(installed, layer_index=4)


def test_shared_forward_passes_none_for_key_and_value(installed):
    attention = _build_attention(installed, layer_index=4)
    hidden_states = torch.arange(4).reshape(1, 4)

    output = attention.forward(torch.tensor([0]), hidden_states)

    assert torch.equal(output, hidden_states)
    assert attention.rotary_emb.args[2] is None
    assert attention.attn.args[1:] == (None, None)


def test_loader_drops_only_redundant_shared_text_weights(installed):
    kept = {
        _ORDINARY_WEIGHT,
        _SHARED_WEIGHT,
        # Tower blocks are named layers.N.self_attn.* too, so an unanchored
        # filter would silently swallow them.
        "vision_tower.encoder.layers.4.self_attn.k_proj.weight",
        "model.audio_tower.encoder.layers.5.self_attn.k_norm.weight",
    }
    dropped = {
        "model.layers.4.self_attn.k_proj.weight",
        "model.layers.4.self_attn.v_proj.weight",
        "model.layers.4.self_attn.k_norm.weight",
        "model.language_model.layers.5.self_attn.k_proj.weight",
    }

    assert _load_weights(installed, kept | dropped, kept) == kept


@pytest.mark.parametrize("missing_weight", [_ORDINARY_WEIGHT, _SHARED_WEIGHT])
def test_genuine_missing_weights_remain_errors(installed, missing_weight):
    required = {_ORDINARY_WEIGHT, _SHARED_WEIGHT}

    with pytest.raises(ValueError, match="Missing weights"):
        _load_weights(installed, required - {missing_weight}, required)


def test_distribution_registers_vllm_general_plugin():
    matching = [
        entry_point
        for entry_point in importlib.metadata.entry_points(group="vllm.general_plugins")
        if entry_point.name == "oumi_gemma4_compat"
    ]

    assert len(matching) == 1
    assert (
        matching[0].value
        == "oumi.utils.vllm_utils.gemma4_compat:register_gemma4_compatibility"
    )
