"""Validate gemma4_kv_share_patch: cache-less forwards must match the cached forward, with and
without gradient checkpointing, and gradients must match a no-checkpoint baseline.

Usage (one free GPU, ~2 min):
  cd experiments/rar_medicine/variant_b && CUDA_VISIBLE_DEVICES=7 python checks/kv_share_patch_check.py
Exit 0 on pass.
"""

import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
BASE = "google/gemma-4-E2B-it"
tok = AutoTokenizer.from_pretrained(BASE)
msgs = [
    {
        "role": "user",
        "content": "A 45-year-old man presents with fever and productive cough. What is the most likely diagnosis?",
    }
]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
resp = (
    "The most likely diagnosis is community-acquired pneumonia. Supporting features are fever, productive cough and focal crackles. "
    * 3
)
ids = tok(text + resp, add_special_tokens=False, return_tensors="pt")[
    "input_ids"
].cuda()
n_prompt = len(tok(text, add_special_tokens=False)["input_ids"])
labels = ids[:, 1:]
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
).cuda()


def resp_logp(logits):
    lg = logits[:, :-1, :].float()
    return (
        F.log_softmax(lg, -1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)[0, n_prompt - 1 :]
    )


def run(use_cache, train, grad_ckpt, need_grad=False):
    model.train(train)
    if grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, use_cache=use_cache)
        lp = resp_logp(out.logits)
        if need_grad:
            (-lp.mean()).backward()
    return lp.detach()


def grad_vec():
    names = [
        "model.language_model.layers.0.self_attn.k_proj.weight",  # feeds a shared layer? (layer 0 is not a producer)
        "model.language_model.layers.14.self_attn.k_proj.weight",  # last non-shared layers are the K/V producers
        "model.language_model.layers.20.self_attn.q_proj.weight",  # a shared layer's own q_proj
        "model.language_model.layers.30.mlp.down_proj.weight",
    ]
    params = dict(model.named_parameters())
    out = {}
    for n in names:
        p = params.get(n)
        out[n.split("layers.")[1]] = (
            None if p is None or p.grad is None else p.grad.float().norm().item()
        )
    return out


print("== before patch")
ref = run(use_cache=True, train=False, grad_ckpt=False)
bad = run(use_cache=False, train=False, grad_ckpt=False)
print(f"  cached eval forward        mean logp {ref.mean():7.3f}")
print(f"  cache-less eval forward    mean logp {bad.mean():7.3f}   (the bug)")

importlib.import_module("gemma4_kv_share_patch")
print("== after patch")
results = {}
for name, kw in {
    "cache-less eval, no ckpt": dict(use_cache=False, train=False, grad_ckpt=False),
    "cache-less train, no ckpt": dict(use_cache=False, train=True, grad_ckpt=False),
    "cache-less train, grad ckpt": dict(use_cache=False, train=True, grad_ckpt=True),
    "cached eval (unchanged path)": dict(use_cache=True, train=False, grad_ckpt=False),
}.items():
    lp = run(**kw)
    d = (lp - ref).abs()
    results[name] = d.max().item()
    print(f"  {name:30s} mean logp {lp.mean():7.3f}   max |d| vs cached {d.max():.4f}")

# gradients: no-ckpt vs ckpt, both cache-less (verl's training forward)
run(use_cache=False, train=True, grad_ckpt=False, need_grad=True)
g_plain = grad_vec()
run(use_cache=False, train=True, grad_ckpt=True, need_grad=True)
g_ckpt = grad_vec()
print(
    "== gradient norms, cache-less training forward (plain vs gradient-checkpointed):"
)
grad_ok = True
for k in g_plain:
    a, b = g_plain[k], g_ckpt[k]
    rel = None if not a else abs(a - b) / a
    print(
        f"  {k:36s} plain {a!s:>12}  ckpt {b!s:>12}  rel diff {rel if rel is None else round(rel, 4)}"
    )
    if a is None or b is None or a == 0 or rel > 0.02:
        grad_ok = False
fwd_ok = all(v < 0.05 for v in results.values())
print(
    "RESULT:",
    "PASS" if fwd_ok and grad_ok else "FAIL",
    f"(forward max|d| {max(results.values()):.4f}, grads {'match' if grad_ok else 'differ'})",
)
sys.exit(0 if fwd_ok and grad_ok else 1)
