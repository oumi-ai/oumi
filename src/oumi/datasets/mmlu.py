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

from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

# MMLU prompts are classified into 57 subjects and 'all' (which contains all subjects)
SUBJECTS = [
    "abstract_algebra",
    "all",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]
SPLITS = [
    "dev",  # For few-shot development; 5 questions per subject (285 questions)
    "validation"  # For selecting hyperparameters (1,531 questions)
    "test",  # For testing purposes (14,042 questions)
]
DEFAULT_NUM_SHOTS = 0  # Values: 0-5; 0 is consistent with LM Evaluation Harness.


# FIXME: Inherit from `BaseMapDataset`.
class MmluDataset:
    # MMLU questions always have 4 possible answers, which are labelled A, B, C, D.
    answer_tokens = ["A", "B", "C", "D"]

    # Static and Class methods for formatting prompts.
    @staticmethod
    def format_subject(subject: str) -> str:
        """Formats the subject of the prompt."""
        return subject.replace("_", " ")

    @classmethod
    def format_example(
        cls, example: dict[str, Any], include_answer: bool = True
    ) -> str:
        """Formats an MMLU example."""
        prompt = example["question"]
        for index, choice in enumerate(example["choices"]):
            prompt += f"\n{cls.answer_tokens[index]}. {choice}"
        prompt += "\nAnswer:"
        if include_answer:
            correct_answer_index = example["answer"]
            prompt += f" {cls.answer_tokens[correct_answer_index]}\n\n"
        return prompt

    @classmethod
    def few_shots(cls, dev_data: Dataset, num_shots: int) -> str:
        """Returns `num_shots` formatted shots from the provided `dev_data`."""
        if not num_shots:
            return ""
        shots: Dataset = dev_data.select(range(num_shots))
        return "".join(
            cls.format_example(example, include_answer=True)  # type: ignore
            for example in shots  # type: ignore
        )

    def __init__(self, subject: str = "all", num_shots: Optional[int] = None):
        """Initializes the class MmluDataset."""
        if subject not in SUBJECTS:
            raise ValueError(f"MMLU: unknown subject `{subject}`")
        self._dataset_dict: DatasetDict = load_dataset("cais/mmlu", subject)  # type: ignore
        self._few_shot_dict: dict[str, str] = dict()
        self._num_shots = num_shots if (num_shots is not None) else DEFAULT_NUM_SHOTS

    # Instance methods (private).
    def _prompt_template(self, example: dict[str, Any]) -> str:
        """Generates the prompt template for evaluations.

        This template is the "original" MMLU implementation by github.com/ollmer.
        For details and undertanding the different options to evaluate with MMLU, see:
        https://huggingface.co/blog/open-llm-leaderboard-mmlu.

        After appying the template, the prompt format will look as follows:
        --------------------------------------------------------------------------------
        The following are multiple choice questions (with answers) about <Subject>.

        <Question-1 from `dev` dataset>
        A. <answer 1.1>
        B. <answer 1.2>
        C. <answer 1.3>
        D. <answer 1.4>
        Answer: B

        <Question-2 from `dev` dataset>
        A. <answer 2.1>
        B. <answer 2.2>
        C. <answer 2.3>
        D. <answer 2.4>
        Answer: A

        [...]

        <Question-5 from `dev` dataset>
        A. <answer 5.1>
        B. <answer 5.2>
        C. <answer 5.3>
        D. <answer 5.4>
        Answer: C

        <Question from `example`>
        A. <answer t.1>
        B. <answer t.2>
        C. <answer t.3>
        D. <answer t.4>
        Answer:
        --------------------------------------------------------------------------------
        """
        subject = example["subject"]
        if subject not in self._few_shot_dict:
            self._update_few_shot_dict(subject)  # Lazy initialization.

        prompt = (
            f"The following are multiple choice questions (with answers) "
            f"about {MmluDataset.format_subject(subject)}.\n\n"
        )
        prompt += self._few_shot_dict[subject]
        prompt += MmluDataset.format_example(example, include_answer=False)
        return prompt

    def _update_few_shot_dict(self, subject: str):
        """Adds few-shot examples for `subject` in the relevant dictionary."""
        assert subject in SUBJECTS
        dataset_dict: DatasetDict = load_dataset("cais/mmlu", subject)  # type: ignore
        dev_dataset: Dataset = dataset_dict["dev"]
        few_shots = MmluDataset.few_shots(dev_dataset, self._num_shots)
        self._few_shot_dict[subject] = few_shots

    def _get_dataset(self, split: str, num_entries: Optional[int] = None) -> Dataset:
        dataset: Dataset = self._dataset_dict[split]
        if num_entries:
            dataset = dataset.select(range(num_entries))
        return dataset

    def _get_formatted_dataset(
        self, split: str, num_entries: Optional[int] = None
    ) -> list[str]:
        dataset: Dataset = self._get_dataset(split=split, num_entries=num_entries)
        dataset_formatted: list[str] = list(map(self._prompt_template, dataset))  # type: ignore
        return dataset_formatted

    def _get_labels(self, split: str, num_entries: Optional[int] = None) -> list[int]:
        dataset: Dataset = self._dataset_dict[split]
        if num_entries:
            dataset = dataset.select(range(num_entries))
        return [example["answer"] for example in dataset]  # type: ignore

    # Instance methods (global).
    # All these will potentially be required by the base `BaseMapDataset`.
    def get_test_split(self, num_entries: Optional[int] = None) -> list[str]:
        """Returns the test split of this dataset."""
        return self._get_formatted_dataset(split="test", num_entries=num_entries)

    def get_validation_split(self, num_entries: Optional[int] = None) -> list[str]:
        """Returns the validation split of this dataset."""
        return self._get_formatted_dataset(split="validation", num_entries=num_entries)

    def get_test_labels(self, num_entries: Optional[int] = None) -> list[int]:
        """Returns the labels of the test dataset."""
        return self._get_labels(split="test", num_entries=num_entries)

    def get_validation_labels(self, num_entries: Optional[int] = None) -> list[int]:
        """Returns the labels of the validation dataset."""
        return self._get_labels(split="validation", num_entries=num_entries)
