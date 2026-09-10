"""Convert a verl FSDP actor checkpoint (model_world_size_N_rank_*.pt) into a
vLLM-loadable HF folder. Thin wrapper over ``verl.model_merger`` that fixes one
Gemma 4 incompatibility: the audio tower has 0-dim buffers (``input_max`` etc.)
that are replicated, not sharded, and ``torch.cat`` on them raises.

    python merge_verl_ckpt.py models/<run>/verl_output/global_step_200/actor models/fullft_step200
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch
from verl.model_merger import fsdp_model_merger
from verl.model_merger.base_model_merger import ModelMergerConfig

_cat = torch.cat


def _safe_cat(tensors, dim=0, **kw):
    """Non-DTensor entries in an FSDP sharded state dict are *replicated* buffers
    (0-dim scalars, 1-element scales, ...): every rank holds the same values, so
    concatenating them would produce a tensor N times too large. Return the
    first copy when all shards are identical; otherwise concatenate as usual.
    """
    if isinstance(tensors, (list, tuple)) and len(tensors) > 1:
        t0 = tensors[0]
        if t0.dim() == 0 or all(
            t.shape == t0.shape and torch.equal(t, t0) for t in tensors[1:]
        ):
            return t0
    return _cat(tensors, dim=dim, **kw)


class _Merger(fsdp_model_merger.FSDPModelMerger):
    """Sharded DTensors always concatenate (even if shards happen to be equal, e.g.
    barely-trained RMSNorm weights); the replicated shortcut in ``_safe_cat`` must
    only apply to the plain-tensor path in ``_load_and_merge_state_dicts``.
    """

    def _merge_by_placement(self, tensors, placement):
        if placement.is_replicate():
            return tensors[0]
        if placement.is_partial():
            raise NotImplementedError("Partial placement is not supported yet")
        return _cat(tensors, dim=placement.dim).contiguous()


def main(local_dir: str, target_dir: str) -> None:
    fsdp_model_merger.torch.cat = (
        _safe_cat  # affects only the non-DTensor cat in the merger module
    )
    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=local_dir,
        target_dir=target_dir,
        hf_model_config_path=str(Path(local_dir) / "huggingface"),
    )
    _Merger(cfg).merge_and_save()
    # vLLM needs the processor config; verl's huggingface/ folder does not ship it.
    src = Path(local_dir).parents[2] / "processor_config.json"
    if src.exists() and not (Path(target_dir) / "processor_config.json").exists():
        shutil.copy(src, target_dir)
    print("done ->", target_dir, sorted(p.name for p in Path(target_dir).iterdir()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
