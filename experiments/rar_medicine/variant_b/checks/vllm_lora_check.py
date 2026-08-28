"""Load the seed LoRA adapter into vLLM for google/gemma-4-E2B-it and check it is a no-op.

Proves, on one free GPU in ~2 min:
  1. vLLM's Gemma4ForConditionalGeneration accepts --enable-lora (needs the site-packages
     patch from GEMMA4_VERL_GRPO_FIXES.md issue 13 / Patch 5; without it this fails at
     engine start with "does not support LoRA yet").
  2. Every module name in the PEFT adapter resolves in vLLM via hf_to_vllm_mapper and the
     qkv_proj / gate_up_proj packing — vLLM's from_local_checkpoint path raises on any
     unexpected module. (verl's tensor-based sync skips that check, so this is the only
     place the mapping is validated strictly.)
  3. Generation is healthy (non-degenerate answers through the model's chat template).
  4. The zero-initialised adapter leaves greedy token ids unchanged.

    ADAPTER_DIR=/tmp/seed_adapter_r16 CUDA_VISIBLE_DEVICES=4 python vllm_lora_check.py

Exit code 0 on PASS. Run after any vllm reinstall and before launching a LoRA run.
"""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

MODEL = "google/gemma-4-E2B-it"
QUESTIONS = [
    "What is the most sensitive imaging modality for a ureteric stone? Answer in one sentence.",
    "Name the first-line antibiotic for uncomplicated cystitis in a non-pregnant woman. One sentence.",
]


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter = os.environ.get(
        "ADAPTER_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_adapter_r16"),
    )
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        for q in QUESTIONS
    ]
    llm = LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=48)
    base = llm.generate(prompts, sp)
    lora = llm.generate(prompts, sp, lora_request=LoRARequest("seed", 1, adapter))
    ok = True
    for b, lo in zip(base, lora):
        same = b.outputs[0].token_ids == lo.outputs[0].token_ids
        nonempty = len(b.outputs[0].token_ids) > 4 and b.outputs[0].text.strip() != ""
        ok &= same and nonempty
        print(
            ("MATCH " if same else "DIFF  ")
            + ("" if nonempty else "[DEGENERATE] ")
            + repr(b.outputs[0].text[:120])
        )
        if not same:
            print("   lora:", repr(lo.outputs[0].text[:120]))
    print("LORA_TEST_RESULT:", "PASS" if ok else "FAIL")
    # sys.exit (not os._exit): vLLM's atexit hooks must run or the engine core is orphaned on the GPU.
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
