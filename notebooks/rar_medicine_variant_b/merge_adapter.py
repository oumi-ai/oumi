"""Turn the downloaded variant_b checkpoint into something vLLM can load.

The S3 ``unmerged_model/model.safetensors`` is the PEFT-wrapped model saved
*without* merging (keys ``base_model.model.model.*`` + ``lora_A/lora_B``), which
vLLM rejects. The clean LoRA adapter for the final step lives at
``models/variant_b/verl_output/global_step_64/actor/lora_adapter/`` (r=16,
alpha=32, language-model q/k/v/o/gate/up/down only). This script loads the base
(``google/gemma-4-E2B-it``, from the adapter config), applies that adapter,
merges, and saves full weights + tokenizer + processor to
``models/variant_b_merged/``.

If ``--src`` has no ``adapter_config.json`` it is assumed to be a full
checkpoint and is symlinked to ``--dst`` instead.

Usage::

    python merge_adapter.py                       # defaults below
    python merge_adapter.py --base google/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (
    HERE
    / "models"
    / "variant_b"
    / "verl_output"
    / "global_step_64"
    / "actor"
    / "lora_adapter"
)
DST = HERE / "models" / "variant_b_merged"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--dst", default=str(DST))
    p.add_argument("--base", default=None, help="Override base_model_name_or_path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if not src.exists():
        raise SystemExit(f"{src} missing; run download_model.py first")

    adapter_cfg = src / "adapter_config.json"
    if not adapter_cfg.exists():
        print(f"{src} is a full checkpoint (no adapter_config.json); linking -> {dst}")
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else None
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
        return

    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
    )

    cfg = json.loads(adapter_cfg.read_text())
    base = args.base or cfg["base_model_name_or_path"]
    print(
        f"LoRA adapter on base {base!r}: r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
        f"targets={cfg.get('target_modules')}"
    )

    # Gemma 4 checkpoints are Gemma4ForConditionalGeneration (text+vision+audio);
    # the adapter keys are rooted at that class (base_model.model.model.language_model...),
    # so load the same architecture the checkpoint declares rather than a CausalLM head.
    arch = AutoConfig.from_pretrained(base).architectures[0]
    import transformers

    cls = getattr(transformers, arch, AutoModelForCausalLM)
    print(f"loading {base} as {cls.__name__} (bf16, cpu)")
    model = cls.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(src))
    model = model.merge_and_unload()
    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, safe_serialization=True)

    tok_src = src if (src / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(dst)
    try:
        AutoProcessor.from_pretrained(base).save_pretrained(
            dst
        )  # vLLM needs processor_config.json
    except Exception as e:  # text-only base
        print(f"no processor saved: {e}")
    (dst / "MERGE_INFO.json").write_text(
        json.dumps({"base": base, "adapter": str(src)}, indent=2)
    )
    print(f"merged -> {dst}")


if __name__ == "__main__":
    main()
