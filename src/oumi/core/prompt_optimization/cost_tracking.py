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

"""Cost tracking and estimation utilities for prompt optimization."""

from dataclasses import dataclass
from typing import Optional

from oumi.utils.logging import logger

# Approximate token costs for popular models ($/1M tokens)
# These are approximate and should be updated regularly
MODEL_COSTS = {
    # OpenAI models
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    # Anthropic models
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Together AI (approximate)
    "meta-llama/Llama-3-70b": {"input": 0.9, "output": 0.9},
    "meta-llama/Llama-3-8b": {"input": 0.2, "output": 0.2},
    # Default for unknown models (local/free)
    "default": {"input": 0.0, "output": 0.0},
}


@dataclass
class CostEstimate:
    """Cost estimate for an optimization run."""

    estimated_input_tokens: int
    """Estimated number of input tokens."""

    estimated_output_tokens: int
    """Estimated number of output tokens."""

    estimated_total_cost: float
    """Estimated total cost in USD."""

    num_train_examples: int
    """Number of training examples."""

    num_val_examples: int
    """Number of validation examples."""

    num_trials: int
    """Number of optimization trials."""

    model_name: str
    """Name of the model being used."""

    notes: list[str]
    """Additional notes or warnings."""

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the estimate.
        """
        return {
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_total_tokens": self.estimated_input_tokens
            + self.estimated_output_tokens,
            "estimated_total_cost_usd": self.estimated_total_cost,
            "num_train_examples": self.num_train_examples,
            "num_val_examples": self.num_val_examples,
            "num_trials": self.num_trials,
            "model_name": self.model_name,
            "notes": self.notes,
        }

    def print_summary(self) -> None:
        """Print a human-readable summary of the cost estimate."""
        logger.info("\n" + "=" * 80)
        logger.info("COST ESTIMATE")
        logger.info("=" * 80)
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Training Examples: {self.num_train_examples}")
        logger.info(f"Validation Examples: {self.num_val_examples}")
        logger.info(f"Optimization Trials: {self.num_trials}")
        logger.info("\nEstimated Token Usage:")
        logger.info(f"  Input tokens:  ~{self.estimated_input_tokens:,}")
        logger.info(f"  Output tokens: ~{self.estimated_output_tokens:,}")
        total_tokens = self.estimated_input_tokens + self.estimated_output_tokens
        logger.info(f"  Total tokens:  ~{total_tokens:,}")

        if self.estimated_total_cost > 0:
            logger.info(f"\nEstimated Cost: ${self.estimated_total_cost:.2f} USD")

            # Add warnings for high costs
            if self.estimated_total_cost > 100:
                logger.warning(
                    "⚠️  HIGH COST WARNING: Estimated cost exceeds $100. "
                    "Consider using a smaller dataset or fewer trials for "
                    "initial testing."
                )
            elif self.estimated_total_cost > 10:
                logger.warning(
                    "⚠️  MODERATE COST: Estimated cost exceeds $10. "
                    "Ensure this is within your budget."
                )
        else:
            logger.info("\nEstimated Cost: $0 (local model or free tier)")

        if self.notes:
            logger.info("\nNotes:")
            for note in self.notes:
                logger.info(f"  • {note}")

        logger.info("=" * 80 + "\n")


def get_model_costs(model_name: str) -> dict[str, float]:  # type: ignore[misc]
    """Get cost per million tokens for a model.

    Args:
        model_name: Name of the model.

    Returns:
        Dictionary with 'input' and 'output' costs per million tokens.
    """
    # Try exact match first
    if model_name in MODEL_COSTS:
        return MODEL_COSTS[model_name]

    # Try partial matches for model families
    model_lower = model_name.lower()

    for model_key, costs in MODEL_COSTS.items():
        if model_key.lower() in model_lower or model_lower in model_key.lower():
            logger.debug(f"Matched model '{model_name}' to cost model '{model_key}'")
            return costs

    # Default to zero cost (local model)
    logger.debug(f"No cost model found for '{model_name}', assuming local/free model")
    return MODEL_COSTS["default"]


def estimate_optimization_cost(
    model_name: str,
    num_train_examples: int,
    num_val_examples: int,
    num_trials: int,
    optimizer: str,
    avg_input_length: int = 100,
    avg_output_length: int = 50,
) -> CostEstimate:
    """Estimate the cost of running prompt optimization.

    Args:
        model_name: Name of the model to use.
        num_train_examples: Number of training examples.
        num_val_examples: Number of validation examples.
        num_trials: Number of optimization trials.
        optimizer: Optimizer name (mipro, gepa, bootstrap).
        avg_input_length: Average input length in tokens.
        avg_output_length: Average output length in tokens.

    Returns:
        CostEstimate object with detailed cost breakdown.
    """
    costs = get_model_costs(model_name)
    notes = []

    # Estimate number of inference calls based on optimizer
    # These are rough estimates based on typical optimizer behavior
    if optimizer == "mipro":
        # MIPRO generates instructions and demos
        # Roughly: num_trials * (train + val evaluations) + instruction generation
        instruction_generation_calls = num_trials * 2  # Generate and refine
        train_eval_calls = num_trials * num_train_examples
        val_eval_calls = num_trials * num_val_examples
        total_calls = instruction_generation_calls + train_eval_calls + val_eval_calls

        notes.append(
            f"MIPRO performs instruction generation and evaluation across "
            f"{num_trials} trials"
        )
        notes.append(
            f"Estimated ~{total_calls:,} inference calls "
            f"({instruction_generation_calls} instruction gen + "
            f"{train_eval_calls:,} train eval + {val_eval_calls:,} val eval)"
        )

    elif optimizer == "gepa":
        # GEPA uses genetic algorithm with population evaluation
        # Roughly: breadth * depth * evaluations
        breadth = num_trials
        depth = 3  # Default depth
        population_size = breadth * depth
        total_calls = population_size * (num_train_examples + num_val_examples)

        notes.append(
            f"GEPA uses genetic algorithm with population size ~{population_size}"
        )
        notes.append(f"Estimated ~{total_calls:,} inference calls")

    elif optimizer == "bootstrap":
        # Bootstrap is simpler - just selects few-shot examples
        # Roughly: num_train_examples + num_val_examples
        total_calls = num_train_examples + num_val_examples * 2

        notes.append("Bootstrap is the most cost-effective optimizer")
        notes.append(f"Estimated ~{total_calls:,} inference calls")

    else:
        # Unknown optimizer, use conservative estimate
        total_calls = num_trials * (num_train_examples + num_val_examples)
        notes.append(f"Using conservative estimate for optimizer '{optimizer}'")

    # Calculate token estimates
    # Instruction generation uses longer prompts
    instruction_tokens_per_call = 500  # Rough estimate for instruction generation
    regular_tokens_per_call = avg_input_length

    # Estimate input tokens
    # Assume 20% of calls are instruction generation (for MIPRO)
    instruction_calls = int(total_calls * 0.2) if optimizer == "mipro" else 0
    regular_calls = total_calls - instruction_calls

    estimated_input_tokens = (
        instruction_calls * instruction_tokens_per_call
        + regular_calls * regular_tokens_per_call
    )

    # Estimate output tokens
    estimated_output_tokens = total_calls * avg_output_length

    # Calculate costs
    input_cost = (estimated_input_tokens / 1_000_000) * costs["input"]
    output_cost = (estimated_output_tokens / 1_000_000) * costs["output"]
    total_cost = input_cost + output_cost

    # Add warning if data is too small
    if num_train_examples < 50:
        notes.append(
            f"⚠️  Training set is small ({num_train_examples} examples). "
            "Consider using more data for better optimization results."
        )

    return CostEstimate(
        estimated_input_tokens=estimated_input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_total_cost=total_cost,
        num_train_examples=num_train_examples,
        num_val_examples=num_val_examples,
        num_trials=num_trials,
        model_name=model_name,
        notes=notes,
    )


def should_warn_about_cost(
    estimate: CostEstimate, threshold: float = 10.0
) -> tuple[bool, Optional[str]]:
    """Check if cost warning should be shown to user.

    Args:
        estimate: Cost estimate.
        threshold: Cost threshold in USD for showing warning.

    Returns:
        Tuple of (should_warn, warning_message).
    """
    if estimate.estimated_total_cost <= 0:
        return False, None

    if estimate.estimated_total_cost > threshold:
        message = (
            f"This optimization run is estimated to cost "
            f"${estimate.estimated_total_cost:.2f} USD. "
            f"This exceeds the threshold of ${threshold:.2f}. "
            f"Consider starting with fewer trials or a smaller dataset for "
            f"initial testing."
        )
        return True, message

    return False, None
