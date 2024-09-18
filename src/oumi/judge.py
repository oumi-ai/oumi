import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import MISSING

from oumi.core.configs import BaseConfig, GenerationConfig, ModelParams, RemoteParams
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.turn import Conversation, Message, Role
from oumi.inference import RemoteInferenceEngine
from oumi.utils.logging import logger


@dataclass
class JudgeParams(BaseParams):
    """Configuration parameters for the judge."""

    template_file: str = MISSING
    """The path to the JSONL file containing prompt templates."""

    attributes: List[str] = MISSING
    """The attributes to judge."""

    request_column_name: str = "request"
    """Name of column that includes the request."""

    context_column_name: str = "context"
    """Name of column that includes the request's context."""

    response_column_name: str = "response"
    """Name of column that includes the response."""


class JudgeAttributeValueType(str, Enum):
    """The type of the attribute."""

    BOOL = "bool"
    """The attribute is a boolean."""

    CATEGORICAL = "categorical"
    """The attribute is a categorical."""

    LIKERT = "likert"
    """The attribute is a Likert scale."""


@dataclass
class JudgeAttribute(BaseParams):
    """Configuration parameters for the judge."""

    name: str = MISSING
    """The name of the attribute."""

    value_type: JudgeAttributeValueType = MISSING
    """The type of the attribute."""

    few_shots: int = MISSING
    """The template to use for the judge."""

    template: str = MISSING
    """The template to use for the judge."""

    attributes: List[str] = MISSING
    """The attributes to judge."""


@dataclass
class JudgeConfig(BaseConfig):
    judge: JudgeParams = field(default_factory=JudgeParams)

    model: ModelParams = field(default_factory=ModelParams)
    """Configuration parameters for the model used in inference."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    """Configuration parameters for text generation during inference."""


class Judge:
    def __init__(self, config: JudgeConfig):
        """Initialize the Judge."""
        self.config = config
        self.inference_engine = self._create_inference_engine()

    def judge(self, prompt: List[Message]) -> Tuple[Optional[str], Optional[str]]:
        """Judge a prompt."""
        conversation = Conversation(messages=prompt)

        if not self.config.generation:
            raise ValueError("Generation config is required.")

        if not self.config.generation.remote_params:
            raise ValueError("Remote params are required.")

        if not self.inference_engine:
            raise ValueError("Inference engine is required.")

        response = self.inference_engine.infer(
            input=[conversation], generation_config=self.config.generation
        )[0]
        return response.messages[-1].content, None

    def _create_inference_engine(self) -> BaseInferenceEngine:
        """Create the inference engine."""
        # TODO: Initialize the appropriate inference engine based on the config
        return RemoteInferenceEngine(self.config.model)

    @staticmethod
    def _extract_bool_answer(full_answer: str) -> Optional[bool]:
        MATCH_PATTERN = r"<answer>.*</answer>"

        if not full_answer:
            logger.error(f"Full Answer ERROR: {full_answer}")
            return None

        answer_match = re.search(MATCH_PATTERN, full_answer)
        if not answer_match:
            logger.error(f"Answer ERROR: {full_answer}")
            return None

        answer = answer_match.group(0).replace("<answer>", "").replace("</answer>", "")

        if answer[:3].lower() == "yes":
            return True
        elif answer[:2].lower() == "no":
            return False
        else:
            logger.error(f"Extraction ERROR: {full_answer}")
            return None

    def _load_prompt_templates(
        self, attribute_name: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Load prompt templates from a JSONL file."""
        prompt_templates = {}
        template_file = Path(self.config.judge.template_file)

        if not template_file.exists():
            raise FileNotFoundError(f"Template file not found: {template_file}")

        with template_file.open("r") as file:
            for line in file:
                template = json.loads(line)
                attribute = template.pop("attribute")
                prompt_templates[attribute] = template["messages"]

        return prompt_templates

    def generate_prompts(self, df_dataset: pd.DataFrame) -> pd.DataFrame:
        """Generate judge prompts for a dataset."""
        for attribute_name in self.config.judge.attributes:
            prompt_template = self._load_prompt_templates(attribute_name)

            prompt_template = self._load_prompt_templates(attribute_name)

            df_dataset[attribute_name] = df_dataset.apply(
                self.generate_judge_prompt,
                args=(
                    prompt_template,
                    self.config.judge.request_column_name,
                    self.config.judge.context_column_name,
                    self.config.judge.response_column_name,
                ),
                axis=1,
            )

        return df_dataset

    @staticmethod
    def generate_judge_prompt(
        row: pd.Series,
        prompt_template: List[Dict[str, str]],
        request_col_name: str,
        context_col_name: str,
        response_col_name: str,
    ) -> str:
        """Replace variables in prompt templates and return as json."""
        content = prompt_template[-1]["content"]
        content = content.replace("$user_input_request", str(row[request_col_name]))
        content = content.replace("$ai_response", str(row[response_col_name]))
        if not context_col_name or pd.isna(row[context_col_name]).any():
            content = content.replace("\n\n$optional_context", "")
        else:
            content = content.replace("$optional_context", str(row[context_col_name]))
        prompt = prompt_template.copy()
        prompt[-1]["content"] = content
        return json.dumps(prompt)


def judge(*args, **kwargs):
    """Judge a dataset."""
    return judge_dataset(*args, **kwargs)


def judge_dataset(
    judge: Judge, dataset: List[dict], attributes: List[str]
) -> List[dict]:
    """Judge a dataset."""
    judged_dataset = []

    for entry in dataset:
        judged_entry = entry.copy()
        for attribute in attributes:
            prompt = [Message(content=entry[f"prompt_{attribute}"], role=Role.USER)]
            response, exception = judge.judge(prompt)

            judged_entry[f"judge_answer_{attribute}"] = response
            judged_entry[f"judge_answer_tf_{attribute}"] = (
                judge._extract_bool_answer(response) if response else None
            )
            judged_entry[f"judge_exception_{attribute}"] = (
                exception if exception else ""
            )

        judged_dataset.append(judged_entry)

    return judged_dataset


def test():
    """Test the judge module."""
    config = JudgeConfig(
        judge=JudgeParams(
            attributes=["helpful", "honest", "safe", "valid"],
            template_file="path/to/your/template_file.jsonl",
        ),
        model=ModelParams(
            model_name="GPT-3.5-turbo",
        ),
        generation=GenerationConfig(
            max_new_tokens=100,
            remote_params=RemoteParams(
                api_url="http://localhost:1234/v1/chat/completions",
                max_retries=2,
            ),
        ),
    )

    # Create a Judge instance
    judge = Judge(config)

    # Create a small test dataset
    dataset = [
        {
            "request": "What is the capital of France?",
            "context": "We are discussing European geography.",
            "response": "The capital of France is Paris.",
        },
        {
            "request": "How do you bake a cake?",
            "context": "We are talking about baking.",
            "response": "To bake a cake, you need flour, eggs, sugar, and butter.",
        },
    ]

    # Create a DataFrame from the dataset
    df_dataset = pd.DataFrame(dataset)

    # Generate prompts
    df_dataset_with_prompts = judge.generate_prompts(df_dataset)

    # Convert DataFrame back to list of dictionaries
    dataset_with_prompts = df_dataset_with_prompts.to_dict("records")

    # Judge the dataset
    judged_dataset = judge_dataset(judge, dataset_with_prompts, config.judge.attributes)

    # Print the results
    for entry in judged_dataset:
        for attribute in config.judge.attributes:
            print(f"Original prompt ({attribute}):", entry[f"prompt_{attribute}"])
            print(f"Judge's answer ({attribute}):", entry[f"judge_answer_{attribute}"])
            print(
                f"Judge's boolean answer ({attribute}):",
                entry[f"judge_answer_tf_{attribute}"],
            )
            print(f"Exception ({attribute}):", entry[f"judge_exception_{attribute}"])
            print()
        print("=" * 50)


# Example usage
if __name__ == "__main__":
    test()
