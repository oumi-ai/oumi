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

import json
from pathlib import Path
from typing import Any

from oumi.core.configs.prompt_config import PromptOptimizationConfig
from oumi.core.prompt_optimization import (
    BaseOptimizer,
    BootstrapFewShotOptimizer,
    BootstrapFewShotWithOptunaOptimizer,
    GepaOptimizer,
    MiproOptimizer,
    OptimizationResult,
    get_metric_fn,
)
from oumi.utils.logging import logger


def _load_dataset(
    dataset_path: str, max_samples: int | None = None, dataset_name: str = "dataset"
) -> list[dict]:
    """Load and validate dataset from JSONL file.

    Args:
        dataset_path: Path to JSONL file.
        max_samples: Maximum number of samples to load.
        dataset_name: Name of the dataset for error messages.

    Returns:
        List of validated dataset examples.

    Raises:
        DatasetValidationError: If dataset is invalid.
    """
    from oumi.core.prompt_optimization.validation import (
        DatasetValidationError,
        validate_dataset_example,
        validate_dataset_file,
    )

    # Validate file exists and is readable
    path = validate_dataset_file(dataset_path, dataset_name)

    data = []
    line_num = 0

    logger.info(f"Loading {dataset_name} from {dataset_path}...")

    with open(path) as f:
        for line in f:
            line_num += 1

            # Skip empty lines
            if not line.strip():
                continue

            # Parse JSON
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetValidationError(
                    f"{dataset_name} line {line_num}: Invalid JSON: {e}\n"
                    f"Line content: {line[:100]}"
                )

            # Validate example format and extract input/output
            try:
                input_text, output_text = validate_dataset_example(
                    example, line_num, dataset_name
                )
                # Store the validated example
                data.append({"input": input_text, "output": output_text})
            except DatasetValidationError as e:
                # Re-raise with context
                raise DatasetValidationError(str(e))

            if max_samples and len(data) >= max_samples:
                logger.info(
                    f"Reached max_samples limit of {max_samples}, stopping load."
                )
                break

    if not data:
        raise DatasetValidationError(
            f"{dataset_name} file is empty or contains no valid examples: "
            f"{dataset_path}"
        )

    logger.info(f"Loaded {len(data)} valid examples from {dataset_name}")

    return data


def _save_results(result: OptimizationResult, output_dir: str) -> None:
    """Save optimization results to disk.

    Args:
        result: OptimizationResult to save.
        output_dir: Directory to save results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save optimized prompt
    prompt_file = output_path / "optimized_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(result.optimized_prompt)
    logger.info(f"Saved optimized prompt to {prompt_file}")

    # Save few-shot examples
    if result.optimized_demos:
        demos_file = output_path / "optimized_demos.jsonl"
        with open(demos_file, "w") as f:
            for demo in result.optimized_demos:
                f.write(json.dumps(demo) + "\n")
        logger.info(f"Saved {len(result.optimized_demos)} demos to {demos_file}")

    # Save hyperparameters
    if result.optimized_hyperparameters:
        params_file = output_path / "optimized_hyperparameters.json"
        with open(params_file, "w") as f:
            json.dump(result.optimized_hyperparameters, f, indent=2)
        logger.info(f"Saved optimized hyperparameters to {params_file}")

    # Save full results
    results_file = output_path / "optimization_results.json"
    results_dict = {
        "final_score": result.final_score,
        "num_trials": result.num_trials,
        "optimized_prompt": result.optimized_prompt,
        "optimized_hyperparameters": result.optimized_hyperparameters,
        "metadata": result.metadata,
        "training_history": result.training_history,
        "optimization_stats": (
            result.optimization_stats.to_dict() if result.optimization_stats else None
        ),
    }
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Saved full results to {results_file}")


def _get_optimizer(config: PromptOptimizationConfig) -> BaseOptimizer:
    """Get the optimizer instance based on config.

    Args:
        config: Prompt optimization configuration.

    Returns:
        Optimizer instance.
    """
    # Get metric function
    metric_fn = get_metric_fn(config.metric, config.custom_metric_path)

    # Create optimizer based on config
    optimizer_name = config.optimization.optimizer.lower()

    if optimizer_name == "mipro":
        return MiproOptimizer(config, metric_fn)
    elif optimizer_name == "gepa":
        return GepaOptimizer(config, metric_fn)
    elif optimizer_name == "bootstrap":
        return BootstrapFewShotOptimizer(config, metric_fn)
    elif optimizer_name == "optuna":
        return BootstrapFewShotWithOptunaOptimizer(config, metric_fn)
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Supported: mipro, gepa, bootstrap, optuna"
        )


def optimize_prompt(config: PromptOptimizationConfig) -> dict[str, Any]:  # type: ignore[misc]
    """Optimize prompts using the specified configuration.

    Args:
        config: Configuration for prompt optimization.

    Returns:
        Dictionary containing optimization results.

    Raises:
        DatasetValidationError: If datasets are invalid.
        ConfigValidationError: If configuration is invalid.
    """
    from oumi.core.prompt_optimization.validation import (
        validate_dataset_split_sizes,
        validate_optimizer_config,
    )

    logger.info("=" * 80)
    logger.info("Starting Prompt Optimization")
    logger.info("=" * 80)
    logger.info(f"Optimizer: {config.optimization.optimizer.upper()}")
    logger.info(f"Metric: {config.metric}")
    logger.info(f"Trials: {config.optimization.num_trials}")

    # Validate optimizer configuration
    validate_optimizer_config(
        config.optimization.optimizer, config.optimization.num_trials
    )

    # Load datasets with validation
    logger.info("\nLoading datasets...")
    train_data = _load_dataset(
        config.train_dataset_path,  # type: ignore[arg-type]
        config.max_training_samples,
        "training dataset",
    )

    if config.val_dataset_path:
        val_data = _load_dataset(
            config.val_dataset_path,  # type: ignore[arg-type]
            config.max_validation_samples,
            "validation dataset",
        )
    else:
        # Split training data if no validation set provided
        logger.info(
            "No validation dataset specified, splitting training data (80/20)..."
        )
        split_idx = int(len(train_data) * 0.8)
        if split_idx == 0 or split_idx == len(train_data):
            raise ValueError(
                f"Training dataset has only {len(train_data)} examples, "
                f"which is too small to split. Please provide at least 10 examples "
                f"or specify a separate validation dataset."
            )
        val_data = train_data[split_idx:]
        train_data = train_data[:split_idx]
        logger.info(
            f"Split complete: {len(train_data)} training, {len(val_data)} validation"
        )

    # Validate split sizes
    validate_dataset_split_sizes(len(train_data), len(val_data))

    logger.info("\nDataset Summary:")
    logger.info(f"  Training examples: {len(train_data)}")
    logger.info(f"  Validation examples: {len(val_data)}")
    logger.info(f"  Total examples: {len(train_data) + len(val_data)}")

    # Estimate and display costs
    from oumi.core.prompt_optimization.cost_tracking import (
        estimate_optimization_cost,
        should_warn_about_cost,
    )

    cost_estimate = estimate_optimization_cost(
        model_name=config.model.model_name,
        num_train_examples=len(train_data),
        num_val_examples=len(val_data),
        num_trials=config.optimization.num_trials,
        optimizer=config.optimization.optimizer,
    )

    # Display cost estimate
    cost_estimate.print_summary()

    # Check if we should warn about high costs
    should_warn, warning_msg = should_warn_about_cost(cost_estimate, threshold=10.0)
    if should_warn:
        logger.warning("\n" + "!" * 80)
        logger.warning("HIGH COST WARNING")
        logger.warning("!" * 80)
        logger.warning(warning_msg)
        logger.warning(
            "\nTo reduce costs, consider:"
            "\n  • Using fewer trials (--optimization.num_trials=20)"
            "\n  • Using a smaller dataset for initial testing"
            "\n  • Using a cheaper model for development"
            "\n  • Using the 'bootstrap' optimizer which is faster and cheaper"
        )
        logger.warning("!" * 80 + "\n")

        # Require user confirmation for very high costs
        if cost_estimate.estimated_total_cost > 50.0:
            logger.warning(
                f"Estimated cost is ${cost_estimate.estimated_total_cost:.2f} "
                f"which is significant."
            )
            logger.warning("Proceeding with optimization...")

    # Get optimizer
    optimizer = _get_optimizer(config)

    # Run optimization
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running {optimizer.get_optimizer_name()} Optimization")
    logger.info(f"{'=' * 80}\n")

    result = optimizer.optimize(train_data, val_data, config.initial_prompt)

    # Check if optimization actually succeeded
    if result.metadata.get("status") == "error":
        logger.error(f"\nOptimization failed: {result.metadata.get('error')}")
        raise RuntimeError(
            f"Optimization failed: {result.metadata.get('error', 'Unknown error')}"
        )

    # Save results
    logger.info("\nSaving optimization results...")
    _save_results(result, config.output_dir)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("Optimization Complete!")
    logger.info("=" * 80)
    logger.info(f"Final Score: {result.final_score:.4f}")
    logger.info(f"Trials Completed: {result.num_trials}")
    if result.optimized_demos:
        logger.info(f"Few-shot Examples: {len(result.optimized_demos)}")

    # Log optimization stats if available
    if result.optimization_stats:
        stats = result.optimization_stats
        logger.info("\nOptimization Statistics:")
        logger.info(f"  Total Time: {stats.get_elapsed_time():.1f}s")
        logger.info(f"  Examples Processed: {stats.num_examples_processed}")
        logger.info(f"  Inference Calls: {stats.num_inference_calls}")
        if stats.num_failed_calls > 0 and stats.num_inference_calls > 0:
            logger.info(
                f"  Failed Calls: {stats.num_failed_calls} "
                f"({(stats.num_failed_calls / stats.num_inference_calls * 100):.1f}%)"
            )
        logger.info(f"  Success Rate: {stats.get_success_rate() * 100:.1f}%")

    logger.info(f"\nResults saved to: {config.output_dir}")
    logger.info("=" * 80 + "\n")

    return {
        "final_score": result.final_score,
        "num_trials": result.num_trials,
        "output_dir": config.output_dir,
        "num_demos": len(result.optimized_demos),
        "optimization_stats": (
            result.optimization_stats.to_dict() if result.optimization_stats else None
        ),
    }
