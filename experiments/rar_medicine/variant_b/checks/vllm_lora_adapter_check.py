"""Check whether vLLM applies a PEFT LoRA adapter to gemma-4 the way HF+PEFT does.

RESULT ON THIS ENV (vllm 0.19.1 + the SupportsLoRA backport, 2026-08-27): it does
NOT. The adapter loads into the right fused modules (qkv_proj / o_proj /
gate_up_proj / down_proj x 35 layers, non-zero lora_B, verified through the LoRA
manager) but generation is bit-identical to the base model - same top-5 first-token
logprobs, same 96-token greedy continuations on 8/8 prompts - even with lora_alpha
scaled 100x. HF+PEFT with the same adapter diverges from the base within a few
tokens on 6/8 prompts. So vLLM LoRA (oumi `model.adapter_model` + VLLM engine, and
verl's rollout LoRA sync, which feeds the same kernels) is a silent no-op for
`Gemma4ForConditionalGeneration` here. GEMMA4_VERL_GRPO_FIXES.md issue 15.

How it works: compare greedy generations

  HF  base      HF  base + adapter        (transformers + peft, bf16)
  vLLM base     vLLM base + LoRARequest   (vllm, bf16)

vLLM-vs-HF numerics differ slightly, so the HF-base vs vLLM-base common-prefix
length is the noise floor. On prompts where the adapter changes the HF output
within MAX_TOKENS, a working vLLM LoRA path must track HF+LoRA *strictly longer*
than vLLM-base does; if vLLM+LoRA == vLLM-base everywhere, LoRA is not applied.

Usage (one free GPU; ~3 min):
  CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn python vllm_lora_adapter_check.py
Exit status 0 = vLLM applies the adapter, 1 = it does not.
"""

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, cast

import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
BASE = "google/gemma-4-E2B-it"
ADAPTER = str(
    _REPO_ROOT
    / "tmp/rar_medicine/variant_b/output/verl_output/global_step_64/actor/lora_adapter"
)
EVAL_JSONL = _REPO_ROOT / "output/rar_medicine_grpo_verl_variant_b/eval/test_1000.jsonl"
N_PROMPTS = int(os.environ.get("N_PROMPTS", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "96"))


def load_prompts() -> list[list[dict]]:
    convs = []
    with EVAL_JSONL.open() as f:
        for line in f:
            c = json.loads(line)
            convs.append(c["messages"])
            if len(convs) == N_PROMPTS:
                break
    return convs


def hf_generate(
    messages_list: list[list[dict]],
) -> tuple[list[list[int]], list[list[int]]]:
    from peft import PeftModel
    from transformers import (  # split: one line trips gitleaks' generic-key rule
        AutoTokenizer,
        Gemma4ForConditionalGeneration,
    )

    tok = AutoTokenizer.from_pretrained(BASE)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
    model.eval()

    def run(model, disable_adapter: bool) -> list[list[int]]:
        outs = []
        for messages in messages_list:
            ids = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to("cuda")
            ctx = model.disable_adapter() if disable_adapter else torch.no_grad()
            with ctx, torch.no_grad():
                gen = model.generate(
                    **ids,
                    max_new_tokens=MAX_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            outs.append(gen[0, ids["input_ids"].shape[1] :].tolist())
        return outs

    lora = run(model, disable_adapter=False)
    base = run(model, disable_adapter=True)
    del model
    torch.cuda.empty_cache()
    return base, lora


def vllm_generate(
    messages_list: list[list[dict]],
) -> tuple[list[list[int]], list[list[int]]]:
    import vllm
    from vllm.lora.request import LoRARequest

    llm = vllm.LLM(
        model=BASE,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=4096,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
    )
    sp = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=42)
    msgs = cast(Any, messages_list)  # vllm's chat() overloads want its own TypedDicts
    base = [list(o.outputs[0].token_ids) for o in llm.chat(msgs, sp)]
    lora = [
        list(o.outputs[0].token_ids)
        for o in llm.chat(msgs, sp, lora_request=LoRARequest("rar_b", 1, ADAPTER))
    ]
    return base, lora


def prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    messages_list = load_prompts()
    print(f"{len(messages_list)} prompts, {MAX_TOKENS} greedy tokens each")
    hf_base, hf_lora = hf_generate(messages_list)
    vl_base, vl_lora = vllm_generate(messages_list)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    rows = []
    print(
        f"{'i':>2} {'hfB~vlB':>8} {'hfL~vlL':>8} {'hfL~vlB':>8} {'hfB~hfL':>8}   (common-prefix tokens)"
    )
    for i, (hb, hl, vb, vl) in enumerate(zip(hf_base, hf_lora, vl_base, vl_lora)):
        r = dict(
            noise_floor=prefix_len(hb, vb),
            lora_match=prefix_len(hl, vl),
            base_vs_hf_lora=prefix_len(hl, vb),
            adapter_effect=prefix_len(hb, hl),
        )
        rows.append(r)
        print(
            f"{i:>2} {r['noise_floor']:>8} {r['lora_match']:>8} {r['base_vs_hf_lora']:>8} {r['adapter_effect']:>8}"
        )
    print("\nfirst prompt, HF+LoRA :", repr(tok.decode(hf_lora[0])[:200]))
    print("first prompt, vLLM+LoRA:", repr(tok.decode(vl_lora[0])[:200]))
    print("first prompt, vLLM base:", repr(tok.decode(vl_base[0])[:200]))

    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    print("\nmedians:", med)
    informative = [r for r in rows if r["adapter_effect"] < MAX_TOKENS]
    identical_to_base = sum(
        prefix_len(vb, vl) == min(len(vb), len(vl)) for vb, vl in zip(vl_base, vl_lora)
    )
    tracks_hf_lora = sum(r["lora_match"] > r["base_vs_hf_lora"] for r in informative)
    ok = bool(informative) and tracks_hf_lora >= max(1, len(informative) // 2)
    print(
        f"\n{len(informative)}/{len(rows)} prompts where the adapter changes the HF output within "
        f"{MAX_TOKENS} tokens; vLLM+LoRA tracks HF+LoRA better than vLLM-base on "
        f"{tracks_hf_lora} of them; vLLM+LoRA output identical to vLLM-base on "
        f"{identical_to_base}/{len(rows)} prompts.\nRESULT: "
        + (
            "OK - vLLM applies the adapter like HF+PEFT"
            if ok
            else "MISMATCH - vLLM LoRA has no effect / does not track HF+PEFT"
        )
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
