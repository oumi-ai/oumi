"""Does verl's remove-padding path give the same log-probs as the padded sdpa path on gemma-4?

verl (dp_actor._forward_micro_batch, use_remove_padding=True, sp=1) unpads the batch with
flash_attn.bert_padding.unpad_input, feeds input_ids (1, nnz) + position_ids (1, nnz) and
NO attention mask, and relies on HF's flash-attention integration to detect the packed
sequences from the position_ids resets (varlen kernels with cu_seqlens). gemma-4 has
sliding-window + full-attention layers and per-layer embeddings, so check it numerically
before flipping `use_remove_padding: True` in train_verl.yaml.

Builds a verl-shaped batch (prompt left-padded to P, response right-padded to R, position
ids = cumsum(mask) - 1) from real RaR-Medicine prompts, then compares per-response-token
log-probs from:
  A. padded forward, sdpa               (what the run used so far)
  B. padded forward, flash_attention_2  (FA2 kernel support for gemma-4)
  C. packed forward, flash_attention_2  (verl's remove-padding branch)

Usage (one free GPU, ~2 min):
  CUDA_VISIBLE_DEVICES=7 PACKED_IMPL=sdpa python rmpad_fa2_numerics_check.py [val.parquet]
PACKED_IMPL selects the kernel for B and C (sdpa | flex_attention | flash_attention_2). RESULT ON THIS
ENV: flash_attention_2 fails outright ("FlashAttention forward only supports head dimension at most
256"; gemma-4 global layers use a larger head_dim), so remove-padding can only work through the
position_ids-based packed masks that transformers builds for sdpa / flex.
Exit 0 if C matches A within bf16 noise: with NOISE_REF=eager the yardstick is the eager-vs-sdpa
padded difference (1.5x its mean and p99); without it, mean |dlogp| < 0.02 and p99 < 0.2.
MEASURED 2026-09-02 (KV_PATCH=1, PACKED_IMPL=sdpa, NOISE_REF=eager): noise floor mean 0.0148 /
p99 0.295; packed-vs-padded mean 0.0139 / p99 0.249 -> PASS. Without the KV patch both paths are
garbage and differ by 0.27 mean (the same forward bug, not a packing problem).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from einops import rearrange
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers import AutoModelForCausalLM, AutoTokenizer
from verl.utils import torch_functional as verl_F

BASE = "google/gemma-4-E2B-it"
_REPO_ROOT = Path(__file__).resolve().parents[4]
PARQUET = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(
        _REPO_ROOT
        / "tmp/rar_medicine/variant_b/output/medqa_gemma4-e2b-it_fullft_tune/verl_datasets/val.parquet"
    )
)
P, R, B = 512, 1024, 8
PACKED_IMPL = os.environ.get(
    "PACKED_IMPL", "flash_attention_2"
)  # kernel used for B and C
if (
    os.environ.get("KV_PATCH", "1") == "1"
):  # gemma-4 needs the shared-KV shim for ANY cache-less forward
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import gemma4_kv_share_patch  # noqa: F401
dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE)

# ---- verl-shaped batch -----------------------------------------------------------------
df = pd.read_parquet(PARQUET)
prompts = []
for i in range(B):
    msgs = df.iloc[i]["prompt"]
    msgs = [dict(m) for m in msgs]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompts.append(tok(text, add_special_tokens=False)["input_ids"])
filler = (
    "The most likely diagnosis is community-acquired pneumonia. The patient presents with fever, "
    "productive cough and focal crackles; a chest radiograph should be obtained and empiric "
    "amoxicillin started if there are no risk factors for resistant organisms. "
)
filler_ids = tok(filler, add_special_tokens=False)["input_ids"]
resp_lens = [96, 160, 240, 333, 420, 512, 700, 1000]
input_ids = torch.full((B, P + R), tok.pad_token_id, dtype=torch.long)
attention_mask = torch.zeros((B, P + R), dtype=torch.long)
response_mask = torch.zeros((B, R), dtype=torch.long)
for i, p_ids in enumerate(prompts):
    p_ids = p_ids[-P:]
    r_ids = (filler_ids * 20)[: resp_lens[i]]
    input_ids[i, P - len(p_ids) : P] = torch.tensor(p_ids)
    attention_mask[i, P - len(p_ids) : P] = 1
    input_ids[i, P : P + len(r_ids)] = torch.tensor(r_ids)
    attention_mask[i, P : P + len(r_ids)] = 1
    response_mask[i, : len(r_ids)] = 1
position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
input_ids, attention_mask, position_ids, response_mask = (
    t.to(dev) for t in (input_ids, attention_mask, position_ids, response_mask)
)
responses = input_ids[:, -R:]
print(
    f"batch {B} x {P + R}, real tokens {int(attention_mask.sum())} "
    f"({100 * attention_mask.sum().item() / attention_mask.numel():.0f} % of padded)"
)


def padded_logprobs(model):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = out.logits[:, -R - 1 : -1, :]
        lp = verl_F.logprobs_from_logits(logits, responses)
        ent = verl_F.entropy_from_logits_with_chunking(
            logits.reshape(-1, logits.shape[-1])
        ).view(B, R)
    return lp.float(), ent.float()


def packed_logprobs(model):
    """Mirror of dp_actor._forward_micro_batch's rmpad branch (sp=1)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        ids_rmpad = ids_rmpad.transpose(0, 1)  # (1, nnz)
        pos_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
        ).transpose(0, 1)  # (1, nnz)
        rolled = torch.roll(ids_rmpad, shifts=-1, dims=1)
        out = model(
            input_ids=ids_rmpad,
            attention_mask=None,
            position_ids=pos_rmpad,
            use_cache=False,
        )
        logits_rmpad = out.logits.squeeze(0)  # (nnz, V)
        lp_rmpad = verl_F.logprobs_from_logits(logits_rmpad, rolled.squeeze(0))
        ent_rmpad = verl_F.entropy_from_logits_with_chunking(logits_rmpad)
        full_lp = pad_input(lp_rmpad.unsqueeze(-1), indices, B, P + R).squeeze(-1)
        full_ent = pad_input(ent_rmpad.unsqueeze(-1), indices, B, P + R).squeeze(-1)
        peak = torch.cuda.max_memory_allocated() / 2**30
    return full_lp[:, -R - 1 : -1].float(), full_ent[:, -R - 1 : -1].float(), peak


def compare(name, a, b):
    m = response_mask.bool()
    d = (a - b).abs()[m]
    print(
        f"  {name:32s} mean |d| {d.mean():.4f}  p99 {d.quantile(0.99):.4f}  max {d.max():.4f}  "
        f"frac>0.1 {(d > 0.1).float().mean():.4f}   (n={int(m.sum())})"
    )
    return d.mean().item(), d.quantile(0.99).item()


model = (
    AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    .to(dev)
    .eval()
)
lp_a, ent_a = padded_logprobs(model)
print(
    f"A sdpa padded: mean logp {lp_a[response_mask.bool()].mean():.3f}, mean entropy {ent_a[response_mask.bool()].mean():.3f}"
)
NOISE_REF = os.environ.get(
    "NOISE_REF"
)  # e.g. "eager": padded forward with another kernel = bf16 noise floor
if NOISE_REF:
    model.set_attn_implementation(NOISE_REF)
    lp_n, ent_n = padded_logprobs(model)
model.set_attn_implementation(PACKED_IMPL)
print(f"packed/B kernel: {PACKED_IMPL}")
lp_b, ent_b = padded_logprobs(model)
torch.cuda.reset_peak_memory_stats()
lp_c, ent_c, peak = packed_logprobs(model)
print("log-prob differences on response tokens:")
if NOISE_REF:
    compare(f"noise floor: {NOISE_REF} padded vs A sdpa", lp_n, lp_a)
compare(f"B {PACKED_IMPL[:6]} padded vs A sdpa padded", lp_b, lp_a)
mean_c, p99_c = compare(f"C {PACKED_IMPL[:6]} packed vs A sdpa padded", lp_c, lp_a)
compare(f"C packed vs B padded ({PACKED_IMPL[:6]})", lp_c, lp_b)
print("entropy differences:")
compare("C packed vs A padded (entropy)", ent_c, ent_a)
print(
    f"packed forward peak memory (incl. 9.6 GiB weights): {peak:.1f} GiB for {int(attention_mask.sum())} tokens"
)
if NOISE_REF:
    d_n = (lp_n - lp_a).abs()[response_mask.bool()]
    ok = mean_c <= 1.5 * d_n.mean().item() and p99_c <= 1.5 * d_n.quantile(0.99).item()
    print(f"pass rule: packed mean/p99 within 1.5x the {NOISE_REF}-vs-sdpa noise floor")
else:
    ok = mean_c < 0.02 and p99_c < 0.2
print(
    "RESULT:",
    "remove-padding path matches padded sdpa within bf16 noise"
    if ok
    else "MISMATCH — do not enable use_remove_padding",
)
sys.exit(0 if ok else 1)
