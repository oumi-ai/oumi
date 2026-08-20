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

import pytest
import torch

from oumi.utils.verl_model_merger import FSDPModelMerger

WORLD_SIZE = 4


@pytest.fixture
def merger() -> FSDPModelMerger:
    # _merge_unsharded_shards uses no instance state, so skip __init__
    # (which loads a HF model config from disk).
    return FSDPModelMerger.__new__(FSDPModelMerger)


def _sharded_rows(rows: int = 8, cols: int = 6) -> tuple[torch.Tensor, list]:
    """A full [rows, cols] tensor and its dim-0 shards, one per rank."""
    full = torch.arange(float(rows * cols)).reshape(rows, cols)
    return full, list(full.chunk(WORLD_SIZE, dim=0))


#
# Ground truth available: the expected shape decides.
#


def test_replicated_zero_dim_keeps_single_copy(merger):
    """A 0-dim buffer, like gemma-4's min/max trackers, has a full copy on
    every rank, and the merger keeps a single one.

    The pre-fix code crashed on these tensors, because torch.cat cannot
    concatenate 0-dim tensors ("zero-dimensional tensor (at position 0)
    cannot be concatenated").
    """
    shards = [torch.tensor(0.5) for _ in range(WORLD_SIZE)]
    merged = merger._merge_unsharded_shards(
        "buf", shards, {"buf": torch.Size([])}, checkpoint_has_dtensors=True
    )
    assert merged.shape == torch.Size([])
    assert merged.item() == 0.5


def test_replicated_shape1_keeps_single_copy_without_warning(merger, caplog):
    """A shape-[1] buffer, like gemma-4's layer_scalar, has an identical copy
    on every rank, and the merger keeps a single one without warning.

    The pre-fix code silently concatenated these into shape [WORLD_SIZE],
    corrupting the export. The corruption only surfaced later, when
    from_pretrained rejected the file with "ckpt: [4] vs model: [1]".
    """
    shards = [torch.tensor([0.73]) for _ in range(WORLD_SIZE)]
    with caplog.at_level("WARNING"):
        merged = merger._merge_unsharded_shards(
            "scalar", shards, {"scalar": torch.Size([1])}, checkpoint_has_dtensors=True
        )
    assert merged.shape == torch.Size([1])
    assert not caplog.records


def test_sharded_tensor_is_concatenated_in_rank_order(merger):
    """Genuine dim-0 shards are concatenated back in rank order.

    This also covers FSDP giving the last rank a smaller slice when sizes
    do not divide evenly: 7 rows over 4 ranks splits as 2+2+2+1.
    """
    full = torch.arange(42.0).reshape(7, 6)
    shards = list(full.chunk(WORLD_SIZE, dim=0))
    merged = merger._merge_unsharded_shards(
        "w", shards, {"w": full.shape}, checkpoint_has_dtensors=False
    )
    assert torch.equal(merged, full)


def test_frozen_constant_shards_are_still_concatenated(merger):
    """All-identical shards of a genuinely sharded tensor must be
    concatenated, not collapsed to one copy.

    Identical values do not imply replication. A frozen all-ones norm
    weight, e.g. produces bit-identical shards on every rank, and
    only the expected shape can tell it apart from a replicated buffer.
    """
    shards = [torch.ones(2, 6) for _ in range(WORLD_SIZE)]
    merged = merger._merge_unsharded_shards(
        "w", shards, {"w": torch.Size([8, 6])}, checkpoint_has_dtensors=False
    )
    assert merged.shape == torch.Size([8, 6])


def test_divergent_replicas_warn_and_keep_rank_zero(merger, caplog):
    """Replicas that differ across ranks, such as unsynced statistics
    buffers, keep the rank-0 copy and emit a warning."""
    shards = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    with caplog.at_level("WARNING"):
        merged = merger._merge_unsharded_shards(
            "buf", shards, {"buf": torch.Size([1])}, checkpoint_has_dtensors=True
        )
    assert merged.item() == 1.0
    assert any("differs across ranks" in r.message for r in caplog.records)


def test_impossible_shape_raises_naming_the_key(merger):
    """When neither one shard nor the dim-0 concatenation matches the
    expected shape, the merger raises an error naming the tensor instead of
    guessing silently."""
    shards = [torch.ones(3, 5) for _ in range(WORLD_SIZE)]
    with pytest.raises(ValueError, match="'w'"):
        merger._merge_unsharded_shards(
            "w", shards, {"w": torch.Size([8, 6])}, checkpoint_has_dtensors=True
        )


def test_zero_dim_with_mismatched_expected_shape_raises(merger):
    """A 0-dim tensor can never be treated as a dim-0 shard. If it does not
    match the expected shape, the merger raises an error, and torch.cat is
    never reached for it."""
    shards = [torch.tensor(0.5) for _ in range(WORLD_SIZE)]
    with pytest.raises(ValueError, match="'buf'"):
        merger._merge_unsharded_shards(
            "buf", shards, {"buf": torch.Size([1])}, checkpoint_has_dtensors=True
        )


#
# No expected shape for the tensor: the merger infers from the checkpoint
# format. A checkpoint that contains DTensors saves plain tensors only for
# replicated buffers, while a legacy checkpoint with no DTensors saves
# plain dim-0 shards.
#


def test_unknown_key_modern_format_keeps_one_copy(merger, caplog):
    """A key the architecture does not declare, such as a custom head, is
    warned about and then merged by checkpoint format. Since this checkpoint
    contains DTensors, its plain tensors are replicated buffers, so one copy
    is kept."""
    shards = [torch.tensor([0.73]) for _ in range(WORLD_SIZE)]
    with caplog.at_level("WARNING"):
        merged = merger._merge_unsharded_shards(
            "custom_head.weight",
            shards,
            {"other": torch.Size([1])},
            checkpoint_has_dtensors=True,
        )
    assert merged.shape == torch.Size([1])
    assert any(
        "not part of the target model architecture" in r.message for r in caplog.records
    )


def test_no_expected_shapes_falls_back_silently_per_key(merger, caplog):
    """When expected_shapes is None, the target model could not be
    instantiated at all. _get_expected_shapes already warned once for the
    whole merge, so no additional per-key warning is emitted here."""
    shards = [torch.tensor(0.5) for _ in range(WORLD_SIZE)]
    with caplog.at_level("WARNING"):
        merged = merger._merge_unsharded_shards(
            "buf", shards, None, checkpoint_has_dtensors=True
        )
    assert merged.shape == torch.Size([])
    assert not caplog.records


def test_fallback_zero_dim_collapses_in_any_format(merger):
    """A 0-dim tensor collapses to one copy in either checkpoint format,
    because a scalar cannot be sharded along dim 0."""
    shards = [torch.tensor(0.5), torch.tensor(0.5)]
    merged = merger._merge_unsharded_shards(
        "buf", shards, None, checkpoint_has_dtensors=False
    )
    assert merged.shape == torch.Size([])


def test_fallback_modern_format_divergent_replicas_warn(merger, caplog):
    """In a checkpoint that contains DTensors, plain tensors are replicated
    buffers. When the copies diverge, the merger keeps rank 0's and warns."""
    shards = [torch.tensor([1.0]), torch.tensor([2.0])]
    with caplog.at_level("WARNING"):
        merged = merger._merge_unsharded_shards(
            "buf", shards, None, checkpoint_has_dtensors=True
        )
    assert merged.item() == 1.0
    assert any("differs across ranks" in r.message for r in caplog.records)


def test_fallback_legacy_format_concatenates_even_identical_shards(merger):
    """In a legacy checkpoint with no DTensors, plain tensors are dim-0
    shards and are always concatenated, exactly as the pre-fix code did.
    That includes bit-identical shards such as frozen constants: value
    equality plays no part in the decision, so the fallback cannot corrupt
    a legacy checkpoint that the old code handled correctly.
    """
    identical = [torch.ones(2, 6) for _ in range(WORLD_SIZE)]
    merged = merger._merge_unsharded_shards(
        "w", identical, None, checkpoint_has_dtensors=False
    )
    assert merged.shape == torch.Size([8, 6])

    full, distinct = _sharded_rows()
    merged = merger._merge_unsharded_shards(
        "w", distinct, None, checkpoint_has_dtensors=False
    )
    assert torch.equal(merged, full)
