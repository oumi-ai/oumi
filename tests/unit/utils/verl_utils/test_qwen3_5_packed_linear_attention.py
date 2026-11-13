# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
from unittest.mock import patch

import pytest
import torch

modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
configuration = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")

_MODULE_NAME = "oumi.utils.verl_utils.qwen3_5_packed_linear_attention"
_FIRST_LENGTH = 5
_SECOND_LENGTH = 7
_VOCAB_SIZE = 64


@pytest.fixture
def install_patch():
    """Imports the patch module fresh and restores the original forwards after."""
    decoder_forward = modeling.Qwen3_5DecoderLayer.forward
    linear_attention_forward = modeling.Qwen3_5GatedDeltaNet.forward

    def install():
        sys.modules.pop(_MODULE_NAME, None)
        return importlib.import_module(_MODULE_NAME)

    yield install
    modeling.Qwen3_5DecoderLayer.forward = decoder_forward
    modeling.Qwen3_5GatedDeltaNet.forward = linear_attention_forward
    if hasattr(
        modeling.Qwen3_5DecoderLayer, "_qwen3_5_packed_linear_attention_patched"
    ):
        del modeling.Qwen3_5DecoderLayer._qwen3_5_packed_linear_attention_patched
    sys.modules.pop(_MODULE_NAME, None)


def _tiny_model():
    torch.manual_seed(0)
    config = configuration.Qwen3_5TextConfig(
        vocab_size=_VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=128,
    )
    config._attn_implementation = "sdpa"
    return modeling.Qwen3_5ForCausalLM(config)


def _sequences():
    generator = torch.Generator().manual_seed(1)
    return (
        torch.randint(0, _VOCAB_SIZE, (_FIRST_LENGTH,), generator=generator),
        torch.randint(0, _VOCAB_SIZE, (_SECOND_LENGTH,), generator=generator),
    )


def _logits(model, input_ids, position_ids, attention_mask=None):
    return model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits


def _separate_logits(model, first, second):
    return torch.cat(
        [
            _logits(model, sequence[None], torch.arange(len(sequence))[None])[0]
            for sequence in (first, second)
        ]
    )


def _packed_position_ids(as_mrope: bool):
    positions = torch.cat([torch.arange(_FIRST_LENGTH), torch.arange(_SECOND_LENGTH)])
    if as_mrope:
        # VERL's packed rows for Qwen VL models: text positions, then 3 rope axes.
        return positions[None, None, :].expand(4, 1, -1)
    return positions[None]


@pytest.mark.parametrize("as_mrope", [False, True])
def test_packed_row_matches_each_sequence_alone(install_patch, as_mrope):
    model = _tiny_model().eval()
    first, second = _sequences()
    packed_ids = torch.cat([first, second])[None]
    position_ids = _packed_position_ids(as_mrope)
    with torch.no_grad():
        separate = _separate_logits(model, first, second)
        leaked = _logits(model, packed_ids, position_ids)[0]
        assert not torch.allclose(leaked, separate, atol=1e-5)

        install_patch()
        packed = _logits(model, packed_ids, position_ids)[0]

    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=1e-5)


def test_packed_gradients_match_under_gradient_checkpointing(install_patch):
    install_patch()
    first, second = _sequences()

    separate_model = _tiny_model().train()
    separate_model.gradient_checkpointing_enable({"use_reentrant": False})
    _separate_logits(separate_model, first, second).sum().backward()

    packed_model = _tiny_model().train()
    packed_model.gradient_checkpointing_enable({"use_reentrant": False})
    _logits(
        packed_model,
        torch.cat([first, second])[None],
        _packed_position_ids(as_mrope=False),
    ).sum().backward()

    for (name, packed), (_, separate) in zip(
        packed_model.named_parameters(), separate_model.named_parameters()
    ):
        torch.testing.assert_close(packed.grad, separate.grad, msg=name)


def test_single_sequence_is_unchanged(install_patch):
    model = _tiny_model().eval()
    first, _ = _sequences()
    position_ids = torch.arange(_FIRST_LENGTH)[None]
    with torch.no_grad():
        before = _logits(model, first[None], position_ids)
        install_patch()
        after = _logits(model, first[None], position_ids)

    torch.testing.assert_close(after, before, atol=0.0, rtol=0.0)


def test_masked_batch_is_unchanged(install_patch):
    model = _tiny_model().eval()
    first, second = _sequences()
    padding = _SECOND_LENGTH - _FIRST_LENGTH
    input_ids = torch.stack(
        [torch.cat([torch.zeros(padding, dtype=torch.long), first]), second]
    )
    attention_mask = torch.stack(
        [
            torch.cat([torch.zeros(padding), torch.ones(_FIRST_LENGTH)]),
            torch.ones(_SECOND_LENGTH),
        ]
    ).long()
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        before = _logits(model, input_ids, position_ids, attention_mask)
        install_patch()
        after = _logits(model, input_ids, position_ids, attention_mask)

    torch.testing.assert_close(after, before, atol=0.0, rtol=0.0)


def test_install_skips_without_qwen3_5(install_patch):
    # A None entry makes importing the module raise ImportError.
    with patch.dict(sys.modules, {modeling.__name__: None}):
        install_patch()

    assert not hasattr(
        modeling.Qwen3_5DecoderLayer, "_qwen3_5_packed_linear_attention_patched"
    )
