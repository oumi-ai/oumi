#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from oumi import train  # noqa: E402
from oumi.core.configs import TrainingConfig  # noqa: E402


def _write_subset(input_path: Path, output_path: Path, rows: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as src, output_path.open("w") as dst:
        for idx, line in enumerate(src):
            if rows and idx >= rows:
                break
            dst.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal GRPO training step with rubric_reward."
    )
    parser.add_argument(
        "--config",
        default="configs/examples/grpo_rlvr/train_weighted.yaml",
        help="Training config path.",
    )
    parser.add_argument(
        "--dataset",
        default="configs/examples/grpo_rlvr/sample_data_weighted.jsonl",
        help="Input dataset path (jsonl).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of dataset rows to use for the smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/rlvr_real/grpo_smoke",
        help="Output directory for the run.",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--enable-wandb", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config)

    subset_path = Path(args.output_dir) / "subset.jsonl"
    _write_subset(Path(args.dataset), subset_path, args.rows)

    config.data.train.datasets[0].dataset_path = str(subset_path)
    config.training.output_dir = args.output_dir
    config.training.max_steps = args.max_steps
    config.training.save_steps = args.max_steps
    config.training.logging_steps = 1
    config.training.enable_wandb = args.enable_wandb
    config.training.enable_tensorboard = False
    config.training.grpo.use_vllm = args.use_vllm
    config.training.grpo.max_completion_length = args.max_completion_length

    if config.training.grpo.num_generations > 2:
        config.training.grpo.num_generations = 2
        config.training.per_device_train_batch_size = 2

    print("Starting GRPO smoke run...")
    print(f"Config: {args.config}")
    print(f"Dataset subset: {subset_path}")
    print(f"Output dir: {args.output_dir}")

    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
