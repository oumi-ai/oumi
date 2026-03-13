# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example: Run AIDE agentic code optimization programmatically.

This script demonstrates how to use Oumi's AIDE integration to automatically
search for optimal training configurations using LLM-powered tree search.

Unlike traditional hyperparameter tuning (``oumi tune``), AIDE operates in
*code space* — it generates entire training scripts, executes them, evaluates
the results, and iteratively improves the code.

Prerequisites:
    pip install oumi[aide]
    pip install aideml --no-deps  # Separate due to version pinning

Usage:
    python scripts/examples/aide/run_aide_optimization.py

    # Or via CLI:
    oumi aide -c configs/recipes/smollm/aide/135m/aide.yaml
"""

from oumi import aide
from oumi.core.configs import (
    AideConfig,
    AideParams,
    AideSearchParams,
    DataParams,
    DatasetParams,
    DatasetSplitParams,
    ModelParams,
)
from oumi.core.configs.params.aide_params import (
    AideExecParams,
    AideLLMParams,
    AideOptimizationSurface,
)


def main():
    """Run a simple AIDE optimization on SmolLM 135M."""
    # Build config programmatically (alternative to YAML)
    config = AideConfig(
        model=ModelParams(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct",
            torch_dtype_str="bfloat16",
            trust_remote_code=True,
        ),
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
                        split="train[:90%]",
                    ),
                ],
            ),
            validation=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="yahma/alpaca-cleaned",
                        split="train[90%:]",
                    ),
                ],
            ),
        ),
        goal=(
            "Optimize training hyperparameters for SmolLM 135M on Alpaca "
            "to minimize eval_loss. Focus on learning rate, optimizer, "
            "and warmup schedule."
        ),
        base_training_config="configs/recipes/smollm/sft/135m/train.yaml",
        mutable_config_paths=[
            "training.learning_rate",
            "training.optimizer",
            "training.warmup_ratio",
        ],
        aide=AideParams(
            steps=5,
            surface=AideOptimizationSurface.CONFIG_SEARCH,
            target_metric="eval_loss",
            target_direction="minimize",
            output_dir="output/aide_example",
            workspace_dir="workspaces/aide_example",
            code_llm=AideLLMParams(model="o4-mini", temperature=0.5),
            feedback_llm=AideLLMParams(model="gpt-4.1-mini", temperature=0.5),
            search=AideSearchParams(num_drafts=2, debug_prob=0.5),
            execution=AideExecParams(timeout=600),
        ),
    )

    # Run optimization
    print("Starting AIDE optimization...")
    print(f"  Surface: {config.aide.surface.value}")
    print(f"  Steps: {config.aide.steps}")
    print(f"  Metric: {config.aide.target_metric} ({config.aide.target_direction})")
    print()

    result = aide(config, verbose=True)

    # Print results
    print("\n" + "=" * 60)
    print("AIDE Optimization Results")
    print("=" * 60)
    print(f"  Best metric: {result.best_metric}")
    print(f"  Total steps: {result.total_steps}")
    print(f"  Good solutions: {result.good_solutions}")
    print(f"  Buggy solutions: {result.buggy_solutions}")
    print(f"  Best solution: {result.best_solution_path}")
    print(f"  Journal: {result.journal_path}")

    if result.best_code:
        print("\nBest solution code (first 500 chars):")
        print("-" * 40)
        print(result.best_code[:500])
        print("-" * 40)


if __name__ == "__main__":
    main()
