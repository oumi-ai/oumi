"""Make gemma-4's KV-shared layers work in cache-less training forwards (HF transformers 5.x).

Why this exists
---------------
gemma-4-E2B declares `num_kv_shared_layers: 20` (of 35): those layers must reuse the K/V of
the last non-shared layer of the same type. transformers implements the reuse through the
cache object only (`Gemma4TextAttention.forward`: `if self.is_kv_shared_layer and
past_key_values is not None: ... shared_layers[...]`, else it computes K/V from the layer's own
k_proj/v_proj, which the checkpoint never trained for that purpose). Any forward without a
cache is therefore wrong: mean response log-prob -12.1 instead of -2.3 and greedy tokens are
noise (checks/kv_share_patch_check.py). verl calls every training forward with
`use_cache=False` (dp_actor._forward_micro_batch), and `GradientCheckpointingLayer.__call__`
strips `past_key_values` under gradient checkpointing anyway, so every verl old-log-prob, ref
and PPO-update forward on gemma-4 was computing garbage (2026-08 runs: actor entropy 11.7 ~
ln(262144), rollout_probs_diff_mean 0.76, Pearson 0.09 vs vLLM, whose own gemma-4 implements
the sharing natively).

What it does
------------
Installs a per-forward shim in place of the missing cache: `Gemma4TextModel.forward` resets a
tiny holder at entry; `Gemma4TextAttention.forward` receives that holder whenever it would have
received `past_key_values=None`. The holder implements only what the attention touches -
`shared_layers` (written by `store_full_length_kv` layers, read by shared layers) and a no-op
`update()` - so no cache bookkeeping, no sequence-length state, no interaction with mask
construction. Real caches (generation) pass through untouched.

Gradient checkpointing (non-reentrant, what verl enables) recomputes a layer during backward;
a shared layer's recompute reads the holder, which still carries the K/V tensors of the
original forward (they are only replaced at the next model forward), so the recomputed ops
match the recorded ones and gradients flow into the producing layer's k_proj/v_proj exactly
as in the cache-based path. Cost: the K/V of the two producing layers stay alive between
forwards (tens of MiB per micro-batch).

How it is loaded
----------------
`actor_rollout_ref.model.external_lib: gemma4_kv_share_patch` in train_verl.yaml; verl's FSDP
workers import that module at init_model (verl.utils.import_utils.import_external_libs) before
touching the model class. The directory is on PYTHONPATH via run.sh. Set
GEMMA4_PATCH_MEMLOG=1 to also print live/peak GPU memory at every text-model forward (rank 0),
which is the cheapest per-phase memory trace available inside verl's workers.
"""

import os

import torch
from transformers.models.gemma4 import modeling_gemma4 as _g4

_MEMLOG = os.environ.get("GEMMA4_PATCH_MEMLOG", "0") == "1"
_GiB = float(2**30)


class _SharedKVHolder:
    """Stand-in for the cache: carries the shared K/V for one forward, nothing else."""

    __slots__ = ("shared_layers",)

    def __init__(self):
        self.shared_layers = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return key_states, value_states

    def get_seq_length(self, layer_idx=0):
        return 0


_holder = _SharedKVHolder()


def _rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


_orig_attn_forward = _g4.Gemma4TextAttention.forward
_orig_text_forward = _g4.Gemma4TextModel.forward


def _attn_forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    # Same signature as Gemma4TextAttention.forward (transformers 5.5.1); only the cache
    # argument is touched.
    if past_key_values is None:
        past_key_values = _holder
    return _orig_attn_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=past_key_values,
        **kwargs,
    )


def _text_forward(self, *args, **kwargs):
    _holder.shared_layers = {}
    if _MEMLOG and _rank0() and torch.cuda.is_available():
        ids = kwargs.get("input_ids", args[0] if args else None)
        shape = tuple(ids.shape) if ids is not None else "?"
        print(
            f"[gemma4-memlog] text forward input {shape} grad={torch.is_grad_enabled()} train={self.training} "
            f"live {torch.cuda.memory_allocated() / _GiB:.2f} GiB peak {torch.cuda.max_memory_allocated() / _GiB:.2f} "
            f"reserved {torch.cuda.memory_reserved() / _GiB:.2f}",
            flush=True,
        )
    return _orig_text_forward(self, *args, **kwargs)


if not getattr(_g4.Gemma4TextAttention, "_kv_share_patched", False):
    _g4.Gemma4TextAttention.forward = _attn_forward
    _g4.Gemma4TextAttention._kv_share_patched = True
    _g4.Gemma4TextModel.forward = _text_forward
    _g4.Gemma4TextModel._kv_share_patched = True
    if _rank0():
        print(
            "[gemma4_kv_share_patch] installed: cache-less forwards now reuse shared K/V",
            flush=True,
        )
