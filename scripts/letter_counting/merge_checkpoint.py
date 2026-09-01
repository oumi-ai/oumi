"""Merge a specific verl FSDP checkpoint into a HuggingFace model directory.

Oumi only auto-merges the *last* checkpoint at the end of training, which for a
run that diverged is the worst one. This exports any step you choose.

Usage:
    python scripts/letter_counting/merge_checkpoint.py \
        --checkpoint output/.../verl_output/global_step_200 \
        --target     output/.../merged_step200
"""

import argparse
import shutil
from pathlib import Path

from oumi.utils.verl_model_merger import FSDPModelMerger, ModelMergerConfig


def main() -> None:
    """Merge the requested verl FSDP checkpoint into a HuggingFace directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional HF model dir to copy preprocessor_config.json from, so "
        "AutoProcessor can load the merged directory.",
    )
    args = parser.parse_args()

    actor_dir = args.checkpoint / "actor"
    if not actor_dir.is_dir():
        raise SystemExit(f"no actor/ dir under {args.checkpoint}")

    FSDPModelMerger(
        ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            tie_word_embedding=False,
            local_dir=str(actor_dir),
            hf_model_config_path=str(actor_dir / "huggingface"),
            target_dir=str(args.target),
        )
    ).merge_and_save()
    print(f"merged {args.checkpoint.name} -> {args.target}")

    if args.base_model:
        src = Path(args.base_model) / "preprocessor_config.json"
        if src.is_file():
            shutil.copy(src, args.target / "preprocessor_config.json")
            print("copied preprocessor_config.json")


if __name__ == "__main__":
    main()
