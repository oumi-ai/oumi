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

import random
from typing import Any, Optional

from oumi.core.prompt_optimization.base import BaseOptimizer, OptimizationResult


class EvolutionaryOptimizer(BaseOptimizer):
    """Evolutionary/genetic algorithm optimizer for prompts.

    **DEPRECATED**: This optimizer is currently not functional and will be replaced
    in a future release. Please use one of the following optimizers instead:

    - 'mipro': Best for large datasets (300+ examples), optimizes instructions and demos
    - 'gepa': Best for complex tasks, uses reflective prompt evolution
    - 'bootstrap': Best for small datasets, optimizes few-shot example selection

    This class is kept for backwards compatibility but will raise NotImplementedError
    if used.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the deprecated evolutionary optimizer."""
        super().__init__(*args, **kwargs)
        # Legacy parameters (not used)
        self.population_size = 10
        self.mutation_rate = 0.3
        self.crossover_rate = 0.5
        self.elite_size = 2

    def get_optimizer_name(self) -> str:
        """Get the optimizer name."""
        return "Evolutionary (DEPRECATED)"

    def _mutate_prompt(self, prompt: str) -> str:
        """Apply mutation to a prompt.

        Args:
            prompt: Input prompt to mutate.

        Returns:
            Mutated prompt.
        """
        mutations = [
            lambda p: f"Please {p.lower()}",
            lambda p: f"{p} Be precise and accurate.",
            lambda p: f"{p} Think step by step.",
            lambda p: f"Task: {p}",
            lambda p: p.replace(".", ". "),
            lambda p: f"{p} Provide a detailed answer.",
            lambda p: f"Expert instruction: {p}",
            lambda p: p.capitalize(),
        ]

        if random.random() < self.mutation_rate:
            mutation = random.choice(mutations)
            return mutation(prompt)
        return prompt

    def _crossover_prompts(self, prompt1: str, prompt2: str) -> str:
        """Perform crossover between two prompts.

        Args:
            prompt1: First parent prompt.
            prompt2: Second parent prompt.

        Returns:
            Offspring prompt.
        """
        if random.random() < self.crossover_rate:
            # Split prompts into sentences
            sentences1 = prompt1.split(". ")
            sentences2 = prompt2.split(". ")

            # Randomly select sentences from each parent
            offspring = []
            for i in range(max(len(sentences1), len(sentences2))):
                if random.random() < 0.5 and i < len(sentences1):
                    offspring.append(sentences1[i])
                elif i < len(sentences2):
                    offspring.append(sentences2[i])

            return ". ".join(offspring)
        return prompt1

    def _evaluate_prompt(self, prompt: str, eval_data: list[dict[str, Any]]) -> float:
        """Evaluate a prompt on a dataset.

        Args:
            prompt: Prompt to evaluate.
            eval_data: Evaluation dataset.

        Returns:
            Score between 0 and 1.
        """
        # Simplified evaluation - in practice, this would use inference engine
        # For now, we score based on prompt characteristics
        score = 0.0

        # Reward length (not too short, not too long)
        length_score = 1.0 - abs(len(prompt) - 100) / 200
        score += max(0, length_score) * 0.3

        # Reward certain keywords
        good_keywords = [
            "step by step",
            "accurate",
            "precise",
            "detailed",
            "expert",
            "think",
        ]
        keyword_score = sum(1 for kw in good_keywords if kw.lower() in prompt.lower())
        score += min(keyword_score / len(good_keywords), 1.0) * 0.4

        # Reward proper capitalization and punctuation
        if prompt and prompt[0].isupper():
            score += 0.15
        if prompt and prompt.endswith("."):
            score += 0.15

        return min(score, 1.0)

    def _select_parents(
        self, population: list[tuple[str, float]], num_parents: int
    ) -> list[str]:
        """Select parents for next generation using tournament selection.

        Args:
            population: List of (prompt, fitness) tuples.
            num_parents: Number of parents to select.

        Returns:
            List of selected parent prompts.
        """
        parents = []
        for _ in range(num_parents):
            # Tournament selection
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        return parents

    def optimize(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        initial_prompt: Optional[str] = None,
    ) -> OptimizationResult:
        """Optimize prompts using evolutionary algorithms.

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            initial_prompt: Optional initial prompt.

        Returns:
            OptimizationResult with optimized prompts and metadata.

        Raises:
            NotImplementedError: This optimizer is deprecated and not functional.
        """
        raise NotImplementedError(
            "The Evolutionary optimizer is currently deprecated and not functional.\n\n"
            "This optimizer was implemented as a prototype but does not perform actual "
            "model-based evaluation. It will be replaced with a proper implementation "
            "in a future release.\n\n"
            "Please use one of these proven optimizers instead:\n\n"
            "  • 'mipro': Best for large datasets (300+ examples)\n"
            "    - Optimizes instructions and few-shot demonstrations\n"
            "    - Uses data-aware instruction generation\n"
            "    Example: --optimization.optimizer=mipro\n\n"
            "  • 'gepa': Best for complex tasks\n"
            "    - Uses reflective prompt evolution with Pareto selection\n"
            "    - Outperforms RL methods by 10%+ with 35x fewer rollouts\n"
            "    Example: --optimization.optimizer=gepa\n\n"
            "  • 'bootstrap': Best for small datasets (10-50 examples)\n"
            "    - Quick few-shot example selection\n"
            "    - Simple and effective for limited data\n"
            "    Example: --optimization.optimizer=bootstrap\n"
        )

        # Rest of the method is kept but never executed
        if self.config.optimization.seed is not None:
            random.seed(self.config.optimization.seed)

        # Initialize population
        base_prompt = initial_prompt or "Answer the following question accurately."
        population = []

        # Create initial population
        for i in range(self.population_size):
            if i == 0:
                prompt = base_prompt
            else:
                prompt = self._mutate_prompt(base_prompt)
            fitness = self._evaluate_prompt(prompt, train_data)
            population.append((prompt, fitness))

        avg_fitness = sum(f for _, f in population) / len(population)
        self._log_progress(f"Initial population fitness: {avg_fitness:.4f}")

        best_fitness_history = []

        # Evolve population
        num_generations = max(
            1, self.config.optimization.num_trials // self.population_size
        )

        for generation in range(num_generations):
            # Sort by fitness
            population.sort(key=lambda x: x[1], reverse=True)

            # Track best fitness
            best_fitness = population[0][1]
            best_fitness_history.append(best_fitness)

            if generation % 5 == 0:
                self._log_progress(
                    f"Generation {generation}/{num_generations}, "
                    f"Best fitness: {best_fitness:.4f}"
                )

            # Elitism - keep best individuals
            new_population = population[: self.elite_size]

            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parents = self._select_parents(population, 2)

                # Crossover
                offspring = self._crossover_prompts(parents[0], parents[1])

                # Mutation
                offspring = self._mutate_prompt(offspring)

                # Evaluate
                fitness = self._evaluate_prompt(offspring, train_data)
                new_population.append((offspring, fitness))

            population = new_population

        # Get best individual
        population.sort(key=lambda x: x[1], reverse=True)
        best_prompt, best_fitness = population[0]

        # Evaluate on validation set
        val_score = self._evaluate_prompt(best_prompt, val_data)

        self._log_progress(
            f"Optimization complete! Best prompt fitness: {best_fitness:.4f}, "
            f"Validation score: {val_score:.4f}"
        )

        return OptimizationResult(
            optimized_prompt=best_prompt,
            optimized_demos=[],
            optimized_hyperparameters={},
            final_score=val_score,
            training_history=[
                {"generation": i, "best_fitness": f}
                for i, f in enumerate(best_fitness_history)
            ],
            num_trials=num_generations * self.population_size,
            metadata={
                "optimizer": "evolutionary",
                "status": "completed",
                "population_size": self.population_size,
                "num_generations": num_generations,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
            },
        )
