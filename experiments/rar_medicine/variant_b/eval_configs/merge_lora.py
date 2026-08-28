"""Merge the verl LoRA adapter into google/gemma-4-E2B-it and save an HF checkpoint.

Why this exists: `oumi train` (VERL_GRPO) exports the final model with
`oumi.utils.verl_model_merger.FSDPModelMerger`, which has no LoRA handling. For a
`lora_rank > 0` run it writes the PEFT-wrapped state dict verbatim
(`base_model.model.<...>.base_layer.weight`, `<...>.lora_A.default.weight`, ...) to
`<output_dir>/model.safetensors` — 2502 tensors that neither transformers nor vLLM
can map onto `Gemma4ForConditionalGeneration`. The adapter verl saved alongside the
checkpoint (`verl_output/global_step_N/actor/lora_adapter/`, standard PEFT format)
is the usable artifact, so this script merges it the standard way:

    base (fp32, CPU)  +  scale * B @ A   ->  bf16 HF checkpoint (+ tokenizer + processor)

The merge is done in fp32 to avoid a second bf16 rounding of the adapter delta
(the adapter tensors are fp32), then cast to bf16 for saving. The processor config
is saved too — verl/HF exports of gemma-4 otherwise lack `processor_config.json` and
vLLM refuses to load the multimodal architecture without it.

Usage:
    python merge_lora.py [--adapter DIR] [--base google/gemma-4-E2B-it] [--out DIR]

Defaults point at the Variant B step-64 adapter and write to
/workspace/persist/shanghong/oumi/tmp/rar_medicine/variant_b/merged_model.
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import torch
from safetensors import safe_open

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_ADAPTER = (
    _REPO_ROOT
    / "tmp/rar_medicine/variant_b/output/verl_output/global_step_64/actor/lora_adapter"
)
_DEFAULT_OUT = _REPO_ROOT / "tmp/rar_medicine/variant_b/merged_model"
_DEFAULT_BASE = "google/gemma-4-E2B-it"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--adapter", type=Path, default=_DEFAULT_ADAPTER)
    p.add_argument("--base", default=_DEFAULT_BASE)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    p.add_argument(
        "--force", action="store_true", help="Overwrite --out if it already exists."
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    adapter_dir: Path = args.adapter
    out_dir: Path = args.out

    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {adapter_dir}")
    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to overwrite it.")
        shutil.rmtree(out_dir)

    from peft import PeftModel
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scale = adapter_cfg["lora_alpha"] / adapter_cfg["r"]
    print(
        f"adapter: {adapter_dir}\n  r={adapter_cfg['r']} alpha={adapter_cfg['lora_alpha']}"
        f" scale={scale} targets={adapter_cfg['target_modules']}"
    )

    t0 = time.time()
    print(f"loading base {args.base} in fp32 on CPU ...")
    base = Gemma4ForConditionalGeneration.from_pretrained(
        args.base, dtype=torch.float32, device_map="cpu"
    )
    print(f"  done in {time.time() - t0:.0f}s")

    # Keep a reference copy of one base weight to verify the merge below.
    probe_name = "model.language_model.layers.0.self_attn.q_proj.weight"
    probe_base = base.get_parameter(probe_name).detach().clone()

    print("attaching adapter ...")
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)

    # Verify every adapter tensor landed on a module (PEFT would otherwise
    # silently leave the module at its base weights).
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
        adapter_keys = list(f.keys())
        probe_a = f.get_tensor(
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        probe_b = f.get_tensor(
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
        )
    model_lora_keys = {
        k.replace(".default", "")
        for k, _ in peft_model.named_parameters()
        if "lora_A" in k or "lora_B" in k
    }
    missing = [k for k in adapter_keys if k not in model_lora_keys]
    extra = sorted(model_lora_keys - set(adapter_keys))
    if missing or extra:
        raise SystemExit(
            f"adapter/model LoRA key mismatch: {len(missing)} adapter tensors not in "
            f"model (e.g. {missing[:3]}), {len(extra)} model LoRA params not in adapter "
            f"(e.g. {extra[:3]})"
        )
    loaded_b = peft_model.get_parameter(
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight"
    )
    if not torch.equal(loaded_b.detach(), probe_b):
        raise SystemExit("lora_B tensor in model differs from adapter file")
    print(f"  all {len(adapter_keys)} adapter tensors matched model LoRA params")

    print("merging ...")
    merged = cast(Any, peft_model).merge_and_unload()
    probe_merged = merged.get_parameter(probe_name).detach()
    expected = probe_base + scale * (probe_b @ probe_a)
    err = (probe_merged - expected).abs().max().item()
    delta = (scale * (probe_b @ probe_a)).abs().max().item()
    print(
        f"  probe {probe_name}: max|merged - (base + scale*BA)| = {err:.3e}, "
        f"max|delta| = {delta:.3e}"
    )
    if err > 1e-5:
        raise SystemExit("merge verification failed")
    if delta == 0.0:
        raise SystemExit(
            "adapter delta is zero on the probe layer; nothing was trained?"
        )

    print(f"casting to bf16 and saving to {out_dir} ...")
    merged = merged.to(torch.bfloat16)
    out_dir.mkdir(parents=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    # Tokenizer + chat template + processor_config.json (needed by vLLM for the
    # multimodal architecture even for text-only use).
    AutoProcessor.from_pretrained(args.base).save_pretrained(str(out_dir))

    (out_dir / "MERGE_INFO.json").write_text(
        json.dumps(
            {
                "base_model": args.base,
                "adapter_dir": str(adapter_dir),
                "adapter_config": adapter_cfg,
                "merge_dtype": "float32",
                "save_dtype": "bfloat16",
                "num_adapter_tensors": len(adapter_keys),
                "probe": {
                    "param": probe_name,
                    "max_abs_err": err,
                    "max_abs_delta": delta,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"done in {time.time() - t0:.0f}s: {sorted(p.name for p in out_dir.iterdir())}"
    )


if __name__ == "__main__":
    main()
