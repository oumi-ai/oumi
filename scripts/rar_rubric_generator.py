#!/usr/bin/env python3
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

r"""RaR (Rubrics as Rewards) Rubric Generation Pipeline.

This script implements the rubric generation methodology from the paper:
"Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains"
(arXiv:2507.17746)

The rubric generation follows four design principles:
1. Grounding in expert references (using reference answers as proxy)
2. Comprehensive coverage across quality dimensions
3. Semantic weighting (Essential, Important, Optional, Pitfall)
4. Self-contained evaluation (no external context needed)

Usage:
    # Generate rubrics for a single prompt
    python scripts/rar_rubric_generator.py --prompt "What causes diabetes?"

    # Generate rubrics from a JSONL file
    python scripts/rar_rubric_generator.py --input data.jsonl \
        --output data_with_rubrics.jsonl

    # Use a specific model
    python scripts/rar_rubric_generator.py --prompt "..." --model gpt-4o

    # Use a reference answer
    python scripts/rar_rubric_generator.py --prompt "..." \
        --reference "Diabetes is caused by..."

Environment:
    OPENAI_API_KEY: Required for OpenAI API access
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)


# Rubric generation prompt template based on the RaR paper methodology
RUBRIC_GENERATION_PROMPT = """You are an expert evaluator creating structured \
rubrics for assessing AI-generated responses.

## Task
Generate a comprehensive set of evaluation rubrics for the following \
question/prompt. The rubrics will be used to train and evaluate language \
models using reinforcement learning.

## Question/Prompt
{prompt}

{reference_section}

## Instructions
Create 7-15 rubric items following these principles:

1. **Comprehensive Coverage**: Cover multiple quality dimensions:
   - Factual accuracy and correctness
   - Completeness of the answer
   - Clarity and organization
   - Appropriate level of detail
   - Logical reasoning
   - Common mistakes to avoid (pitfalls)

2. **Semantic Weighting**: Assign each rubric one of these categories:
   - **Essential** (weight: 5): Core requirements that MUST be satisfied
   - **Important** (weight: 4): Significant points that should be included
   - **Optional** (weight: 2): Helpful additions that improve quality
   - **Pitfall** (weight: -1): Common mistakes to AVOID (e.g., "Avoids X")

3. **Self-Contained**: Each rubric should be evaluatable without external references

4. **Binary Evaluation**: Each rubric should be answerable with yes (1) or no (0)

## Output Format
Return a JSON array of rubric objects. Each object must have:
- "title": Short name (2-4 words)
- "description": One sentence stating what to evaluate (start with category name)
- "weight": Numeric weight (5, 4, 2, or -1)

Example format:
```json
[
  {{"title": "Core", "description": "Essential: Defines concept.", "weight": 5}},
  {{"title": "Evidence", "description": "Important: Has examples.", "weight": 4}},
  {{"title": "Structure", "description": "Optional: Well-organized.", "weight": 2}},
  {{"title": "No Errors", "description": "Pitfall: No mistakes.", "weight": -1}}
]
```

Return ONLY the JSON array, no other text."""

REFERENCE_SECTION_TEMPLATE = """## Reference Answer (Use as expert guidance)
{reference}
"""


@dataclass
class Rubric:
    """A single rubric item."""

    title: str
    description: str
    weight: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "title": self.title,
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rubric":
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            description=data.get("description", ""),
            weight=int(data.get("weight", 1)),
        )


@dataclass
class RubricGenerationResult:
    """Result of rubric generation."""

    prompt: str
    rubrics: list[Rubric]
    reference_answer: str | None = None
    model_used: str = "gpt-4o-mini"
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format compatible with RaR dataset."""
        return {
            "question": self.prompt,
            "rubric": [r.to_dict() for r in self.rubrics],
            "rubric_count": len(self.rubrics),
            "reference_answer": self.reference_answer or "",
            "question_source": f"generated_by_{self.model_used}",
        }


class RubricGenerator:
    """Generate rubrics for prompts using the RaR methodology."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        """Initialize the rubric generator.

        Args:
            model: OpenAI model to use (gpt-4o-mini, gpt-4o, o3-mini).
            temperature: Sampling temperature.
            max_retries: Number of retries on failure.
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI()

    def generate(
        self,
        prompt: str,
        reference_answer: str | None = None,
    ) -> RubricGenerationResult:
        """Generate rubrics for a single prompt.

        Args:
            prompt: The question/prompt to generate rubrics for.
            reference_answer: Optional reference answer to guide rubric generation.

        Returns:
            RubricGenerationResult with generated rubrics.
        """
        # Build the reference section if provided
        reference_section = ""
        if reference_answer:
            reference_section = REFERENCE_SECTION_TEMPLATE.format(
                reference=reference_answer
            )

        # Build the full prompt
        full_prompt = RUBRIC_GENERATION_PROMPT.format(
            prompt=prompt,
            reference_section=reference_section,
        )

        # Try to generate rubrics
        for attempt in range(self.max_retries):
            try:
                sys_msg = "You are an expert at creating evaluation rubrics."
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": full_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=2000,
                )

                # Parse the response
                content = response.choices[0].message.content
                rubrics = self._parse_rubrics(content)

                if rubrics:
                    return RubricGenerationResult(
                        prompt=prompt,
                        rubrics=rubrics,
                        reference_answer=reference_answer,
                        model_used=self.model,
                        success=True,
                    )

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return RubricGenerationResult(
                        prompt=prompt,
                        rubrics=[],
                        reference_answer=reference_answer,
                        model_used=self.model,
                        success=False,
                        error=str(e),
                    )

        return RubricGenerationResult(
            prompt=prompt,
            rubrics=[],
            reference_answer=reference_answer,
            model_used=self.model,
            success=False,
            error="Failed to parse rubrics after retries",
        )

    def _parse_rubrics(self, content: str) -> list[Rubric]:
        """Parse rubrics from LLM response.

        Args:
            content: Raw response content.

        Returns:
            List of Rubric objects.
        """
        # Try to extract JSON from the response
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [
                    Rubric.from_dict(item) for item in data if isinstance(item, dict)
                ]
        except json.JSONDecodeError:
            # Try to find JSON array in the content
            import re

            match = re.search(r"\[[\s\S]*\]", content)
            if match:
                try:
                    data = json.loads(match.group())
                    return [
                        Rubric.from_dict(item)
                        for item in data
                        if isinstance(item, dict)
                    ]
                except json.JSONDecodeError:
                    pass

        return []

    def generate_batch(
        self,
        prompts: list[dict[str, Any]],
        show_progress: bool = True,
    ) -> list[RubricGenerationResult]:
        """Generate rubrics for multiple prompts.

        Args:
            prompts: List of dicts with 'prompt' and optional 'reference_answer'.
            show_progress: Whether to show progress.

        Returns:
            List of RubricGenerationResult objects.
        """
        results = []
        total = len(prompts)

        for i, item in enumerate(prompts):
            if show_progress:
                print(f"Generating rubrics [{i + 1}/{total}]...", end="\r")

            prompt = item.get("prompt", item.get("question", ""))
            reference = item.get("reference_answer", item.get("answer", None))

            result = self.generate(prompt, reference)
            results.append(result)

        if show_progress:
            print(f"Generated rubrics for {total} prompts.        ")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation rubrics using the RaR methodology"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Single prompt to generate rubrics for",
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Reference answer to guide rubric generation",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSONL file with prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    generator = RubricGenerator(
        model=args.model,
        temperature=args.temperature,
    )

    if args.prompt:
        # Single prompt mode
        print(f"Generating rubrics for prompt: {args.prompt[:100]}...")
        result = generator.generate(args.prompt, args.reference)

        if result.success:
            print(f"\nGenerated {len(result.rubrics)} rubrics:\n")
            for i, rubric in enumerate(result.rubrics, 1):
                weight_label = {
                    5: "Essential",
                    4: "Important",
                    2: "Optional",
                    -1: "Pitfall",
                }.get(rubric.weight, "Unknown")
                print(f"{i}. [{weight_label}] {rubric.title}")
                print(f"   {rubric.description}")
                print()

            # Output JSON format
            print("\nJSON format:")
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Error: {result.error}")
            sys.exit(1)

    elif args.input:
        # Batch mode
        if not args.output:
            print("Error: --output required when using --input")
            sys.exit(1)

        print(f"Loading prompts from {args.input}...")
        prompts = []
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))

        print(f"Loaded {len(prompts)} prompts")
        results = generator.generate_batch(prompts)

        # Write results
        success_count = sum(1 for r in results if r.success)
        print(
            f"Successfully generated rubrics for {success_count}/{len(results)} prompts"
        )

        with open(args.output, "w") as f:
            for result in results:
                if result.success:
                    f.write(json.dumps(result.to_dict()) + "\n")

        print(f"Results written to {args.output}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
