"""Reproduce verl's step-1 base-weight sync into a LoRA-enabled vLLM engine for gemma-4.

With --enable-lora, vLLM wraps LoRA-capable linears so their parameters become
`<module>.base_layer.weight`. verl's first sync (rollout.load_format dummy_dtensor) streams
the full HF state dict and renames only q/k/v/o/gate/up/down_proj to the `.base_layer` form
(verl/utils/fsdp_utils.py replace_lora_wrapper). Any other wrapped module — gemma-4 has
per_layer_input_gate, per_layer_projection, per_layer_model_projection, embed_tokens,
lm_head — makes vLLM's load_weights raise KeyError, and the trainer then hangs forever.

This check starts vLLM exactly as verl does for LoRA (enable_lora, max_lora_rank 16, and the
`lora_target_modules` restriction from train_verl.yaml), then on the worker:
  1. lists which parameters vLLM actually wrapped (must be only the 4 target suffixes), and
  2. runs model.load_weights() over the real HF checkpoint with verl's key renaming applied —
     the same call verl's receiver makes — and reports the set of names it loaded.

    LORA_TARGET_MODULES=qkv_proj,o_proj,gate_up_proj,down_proj CUDA_VISIBLE_DEVICES=4 \
        python vllm_base_sync_check.py

Exit 0 = every checkpoint tensor resolved. Run before launching a LoRA run on a new vllm/verl.
"""

import os
import sys
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# collective_rpc with a Python callable needs pickle fallback in vllm 0.19 (msgpack refuses functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

MODEL = "google/gemma-4-E2B-it"
TARGETS = [
    m
    for m in os.environ.get(
        "LORA_TARGET_MODULES", "qkv_proj,o_proj,gate_up_proj,down_proj"
    ).split(",")
    if m
]


def _worker_check(self, ckpt_dir: str) -> dict:
    """Runs inside the vLLM worker process (collective_rpc)."""
    import glob
    import traceback

    from peft import LoraConfig, TaskType
    from safetensors import safe_open
    from verl.utils.fsdp_utils import replace_lora_wrapper

    model = self.model_runner.model
    names = [n for n, _ in model.named_parameters()]
    wrapped = sorted(
        {
            n.rsplit(".base_layer.", 1)[0].split(".")[-1]
            for n in names
            if ".base_layer." in n
        }
    )
    unwrapped_linears = sorted(
        {
            n.split(".")[-2]
            for n in names
            if n.endswith(".weight")
            and ".base_layer." not in n
            and any(
                k in n
                for k in (
                    "per_layer_input_gate",
                    "per_layer_projection",
                    "per_layer_model_projection",
                    "embed_tokens",
                    "lm_head",
                )
            )
        }
    )

    # Same LoraConfig verl builds; replace_lora_wrapper only needs target/exclude fields.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        exclude_modules="(.*vision_tower.*)|(.*audio_tower.*)",
        bias="none",
    )

    def weights():
        for shard in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    yield replace_lora_wrapper(k, peft_config), f.get_tensor(k)

    result: dict[str, Any] = {
        "wrapped_suffixes": wrapped,
        "unwrapped_gemma4_extras": unwrapped_linears,
    }
    try:
        loaded = model.load_weights(weights())
        loaded = set(loaded or [])
        result["loaded_count"] = len(loaded)
        result["loaded_base_layer_count"] = sum(
            1 for n in loaded if ".base_layer." in n
        )
        result["sample_loaded"] = sorted(loaded)[:3]
        result["error"] = None
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-1500:]
    return result


def main() -> None:
    from huggingface_hub import snapshot_download
    from vllm import LLM

    ckpt = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
    print("checkpoint:", ckpt, "| lora_target_modules:", TARGETS, flush=True)
    llm = LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        lora_target_modules=TARGETS,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
    )
    res = llm.collective_rpc(_worker_check, args=(ckpt,))[0]
    print("vLLM wrapped (LoRA) module suffixes:", res["wrapped_suffixes"])
    print("gemma-4 extra linears left unwrapped:", res["unwrapped_gemma4_extras"])
    if res["error"]:
        print("load_weights FAILED:", res["error"])
        print(res.get("traceback", ""))
        print("BASE_SYNC_RESULT: FAIL")
        sys.exit(1)
    print(
        f"load_weights OK: {res['loaded_count']} params loaded, {res['loaded_base_layer_count']} via .base_layer; e.g. {res['sample_loaded']}"
    )
    extra_wrapped = set(res["wrapped_suffixes"]) - set(TARGETS)
    print(
        "BASE_SYNC_RESULT:",
        "PASS"
        if not extra_wrapped
        else f"FAIL unexpected wrapped modules {sorted(extra_wrapped)}",
    )
    # sys.exit (not os._exit): vLLM's atexit hooks must run or the engine core is orphaned on the GPU.
    sys.exit(0 if not extra_wrapped else 1)


if __name__ == "__main__":
    main()
