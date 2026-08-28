"""Build a zero-initialised LoRA adapter for google/gemma-4-E2B-it with the exact
LoraConfig verl constructs from train_verl.yaml (r=16, alpha=32, q/k/v/o/gate/up/down,
vision and audio towers excluded, bias none, task CAUSAL_LM).

PEFT initialises lora_B to zero, so the adapter is numerically a no-op; it exists so
vllm_lora_check.py can prove that (1) vLLM's gemma-4 class accepts --enable-lora and
(2) every adapter module name resolves in vLLM. Runs on CPU (bf16, ~11 GiB RAM, ~1 min).

    ADAPTER_DIR=/tmp/seed_adapter_r16 python seed_adapter.py
"""

import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

MODEL = "google/gemma-4-E2B-it"
# Keep in sync with actor_rollout_ref.model.* in ../train_verl.yaml.
LORA = LoraConfig(
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


def main() -> None:
    out = os.environ.get(
        "ADAPTER_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_adapter_r16"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    print("hf class:", type(model).__name__)
    peft_model = get_peft_model(model, LORA)
    peft_model.save_pretrained(out)

    import safetensors.torch as st

    sd = st.load_file(os.path.join(out, "adapter_model.safetensors"))
    towers = [k for k in sd if "vision_tower" in k or "audio_tower" in k]
    zero_b = all(float(v.abs().max()) == 0 for k, v in sd.items() if "lora_B" in k)
    print(
        f"adapter tensors: {len(sd)} (expect 490 = 35 layers x 7 modules x A/B) | tower tensors: {len(towers)} (expect 0)"
    )
    print(f"sample key: {next(iter(sd))}")
    print(f"all lora_B zero: {zero_b}")
    print("saved to", out)
    assert len(sd) == 490 and not towers and zero_b


if __name__ == "__main__":
    main()
