"""Diagnostic external_lib: dump what verl's actor sees and computes in compute_log_prob.

Loaded via `actor_rollout_ref.model.external_lib: [gemma4_kv_share_patch, verl_logprob_dump_hook]`
(diagnostic runs only). Wraps DataParallelPPOActor.compute_log_prob so that every call saves, per
rank, the input batch tensors (input_ids, attention_mask, position_ids, responses, rollout_log_probs
when present) and the returned log_probs / entropys to
`$VERL_LOGPROB_DUMP_DIR/rank<r>_call<n>.pt`. Offline, compare against a per-sequence HF forward to
find which sequences / tokens the in-run actor scores differently from vLLM.
"""

import os

import torch

_DIR = os.environ.get("VERL_LOGPROB_DUMP_DIR")
_MAX_CALLS = int(os.environ.get("VERL_LOGPROB_DUMP_MAX_CALLS", "2"))

if _DIR:
    from verl.workers.actor import dp_actor

    os.makedirs(_DIR, exist_ok=True)
    _orig = dp_actor.DataParallelPPOActor.compute_log_prob
    _calls = {"n": 0}

    def compute_log_prob(self, data, calculate_entropy=False):
        out = _orig(self, data, calculate_entropy=calculate_entropy)
        if _calls["n"] < _MAX_CALLS:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else 0
            )
            keep = {}
            for k in (
                "input_ids",
                "attention_mask",
                "position_ids",
                "responses",
                "response_mask",
                "rollout_log_probs",
            ):
                if k in data.batch.keys():
                    # clone: verl's batch tensors are views of shared storage, which torch.save rejects
                    keep[k] = data.batch[k].detach().cpu().clone().contiguous()
            keep["log_probs"] = out["log_probs"].detach().float().cpu().clone()
            if "entropys" in out:
                keep["entropys"] = out["entropys"].detach().float().cpu().clone()
            keep["meta_info"] = {
                k: v
                for k, v in data.meta_info.items()
                if isinstance(v, (int, float, str, bool))
            }
            keep["role"] = type(self.actor_module).__name__
            path = os.path.join(_DIR, f"rank{rank}_call{_calls['n']}.pt")
            torch.save(keep, path)
            print(
                f"[logprob-dump] rank {rank} saved {path} ({tuple(keep['log_probs'].shape)})",
                flush=True,
            )
        _calls["n"] += 1
        return out

    dp_actor.DataParallelPPOActor.compute_log_prob = compute_log_prob
    print(f"[verl_logprob_dump_hook] installed, dumping to {_DIR}", flush=True)
