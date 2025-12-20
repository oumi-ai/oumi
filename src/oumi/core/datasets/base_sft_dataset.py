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

import re
from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import pandas as pd
from typing_extensions import override

from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.tokenizers.utils import (
    tokenize_for_completions_only_training_with_prefix,
    tokenize_for_completions_only_training_with_template,
)
from oumi.core.types.conversation import Conversation
from oumi.utils.logging import logger


class BaseSftDataset(BaseMapDataset, ABC):
    """In-memory dataset for SFT data."""

    default_dataset = None

    def __init__(
        self,
        *,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
        split: str | None = None,
        tokenizer: BaseTokenizer | None = None,
        task: Literal["sft", "generation", "auto"] = "auto",
        return_tensors: bool = False,
        text_col: str = "text",
        assistant_only: bool = False,
        response_template: str | None = None,
        instruction_template: str | None = None,
        return_conversations: bool = False,
        return_conversations_format: Literal["dict", "json"] = "json",
        **kwargs,
    ) -> None:
        """Initializes a new instance of the BaseSftDataset class."""
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split=split,
            **kwargs,
        )

        self._task = task
        self._text_col = text_col
        self._tokenizer = tokenizer
        self._return_tensors = "pt" if return_tensors else None

        self._assistant_only = assistant_only
        self._response_template = response_template
        self._instruction_template = instruction_template
        self._return_conversations = return_conversations
        self._return_conversations_format = return_conversations_format

        if self._assistant_only:
            self._verify_assistant_only_compatibility()

        # Only load data if not already set by subclass
        # (subclasses may set _data before calling super().__init__)
        if self._data is None and self._raw_hf_data is None:
            # Check if _load_data is overridden by subclass (legacy path)
            if type(self)._load_data is not BaseMapDataset._load_data:
                self._data = self._load_data()
            else:
                # Use new HF native loading
                self._raw_hf_data = self._load_raw_hf_dataset()

    #
    # Properties
    #
    @property
    def text_col(self) -> str:
        """Gets the text target column.

        The generated text will be stored in this column.
        """
        return self._text_col

    @property
    def task(self) -> str:
        """Gets the task mode for the dataset.

        The generated prompt is often different for generation vs SFT tasks.
        """
        return self._task

    @property
    def assistant_only(self) -> bool:
        """Gets whether the dataset is set to train only on assistant turns."""
        return self._assistant_only

    #
    # Main API
    #

    def prompt(self, idx: int) -> str:
        """Returns the prompt at the specified index.

        Args:
            idx (int): The index of the prompt to retrieve.

        Returns:
            str: The prompt at the specified index.
        """
        return self.tokenize(self.conversation(idx), tokenize=False)[self.text_col]

    def conversation(self, idx: int) -> Conversation:
        """Returns the conversation at the specified index.

        Args:
            idx (int): The index of the conversation to retrieve.

        Returns:
            str: The conversation at the specified index.
        """
        sample = self.raw(idx)
        return self.transform_conversation(sample)

    def conversations(self) -> list[Conversation]:
        """Returns a list of all conversations."""
        indexes = range(len(self))
        return [self.conversation(index) for index in indexes]

    #
    # Abstract Methods
    #
    @abstractmethod
    def transform_conversation(self, example: dict | pd.Series) -> Conversation:
        """Preprocesses the inputs of the example and returns a dictionary.

        Args:
            example (dict): The example containing the input and instruction.

        Returns:
            dict: The preprocessed inputs as a dictionary.

        """
        raise NotImplementedError

    #
    # Pre-processing
    #
    @override
    def transform(self, sample: pd.Series) -> dict:
        """Preprocesses the inputs in the given sample."""
        conversation = self.transform_conversation(sample)
        if self._return_conversations:
            # This may require `use_torchdata=True` for TRL_SFT trainer,
            # but compatible with TRL_GRPO trainer.
            if self._return_conversations_format == "json":
                conversation_json = conversation.to_json()
                return {"conversation_json": conversation_json}
            elif self._return_conversations_format == "dict":
                return conversation.to_dict()
            else:
                raise ValueError(
                    f"Invalid return_conversations_format: "
                    f"'{self._return_conversations_format}'."
                    "Supported formats are 'json' and 'dict'."
                )
        return self.tokenize(conversation)

    def tokenize(
        self,
        sample: dict | pd.Series | Conversation,
        tokenize: bool = True,
    ) -> dict:
        """Applies the chat template carried by the tokenizer to the input example.

        Args:
            sample (Dict): Mapping `messages` to a List containing the (ordered)
                messages exchanged within a single chat dialogue.
                Each item of example["messages"] is a dict mapping the `content` of the
                message and the `role` of the one relayed it.
                E.g., role == 'user' or role == 'assistant'.
            tokenize (bool): Whether to tokenize the messages or not.

        Raises:
            NotImplementedError: Currently only the `sft` task mode is supported.
            ValueError: if requested `task` is not in "sft" or "generation"

        Returns:
            Dict: It adds a `text` key in the input `example` dictionary, mapped to
            a string carrying the `messages` to the tokenizer's chat format.
        """
        if self._tokenizer is None:
            raise ValueError("Tokenizer is required for tokenization.")

        if isinstance(sample, Conversation):
            conversation = sample
        else:
            if isinstance(sample, pd.Series):
                sample = sample.to_dict()

            if isinstance(sample, dict) and "messages" in sample:
                conversation = Conversation.from_dict(sample)

            else:
                raise ValueError(
                    "Input samples must be a Conversation or a dict with "
                    "'messages' key."
                )

        if not self._assistant_only or not tokenize:
            return self._tokenize(conversation, tokenize)

        if self._is_template_compatible_with_completions_only_training:
            return tokenize_for_completions_only_training_with_template(
                tokenizer=self._tokenizer,
                conversation=conversation,
            )
        else:
            return tokenize_for_completions_only_training_with_prefix(
                tokenizer=self._tokenizer,
                conversation=conversation,
                response_template=cast(str, self._response_template),
                instruction_template=cast(str, self._instruction_template),
                response_token_ids=self.response_token_ids,
                instruction_token_ids=self.instruction_token_ids,
            )

    def _tokenize(
        self, sample: dict | pd.Series | Conversation, tokenize: bool = True
    ) -> dict:
        if self._tokenizer is None:
            raise ValueError("Tokenizer is required for tokenization.")

        results = self._tokenizer.apply_chat_template(
            sample,  # type: ignore
            tokenize=tokenize,
            return_dict=tokenize,
            return_tensors=self._return_tensors,
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            add_generation_prompt=(self.task == "generation"),
        )

        if tokenize:
            return cast(dict, results)
        else:
            return {
                self.text_col: results,
            }

    def _verify_assistant_only_compatibility(self) -> None:
        if self._tokenizer is None:
            raise ValueError(
                "Tokenizer is required to enable tokenization "
                "for training on assistant-only turns."
            )

        if self._tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer must have a chat template to enable "
                "tokenization for training on assistant-only turns."
            )

        template: str = self._tokenizer.chat_template  # type: ignore

        if re.search(r"\{\%-?\s*generation\s*-?\%\}", template):
            logger.info(
                "Tokenizer template contains `{% generation %}` keyword. "
                "We will use it for completions-only training."
            )

            self._is_template_compatible_with_completions_only_training = True
        else:
            if (
                self._response_template is None
                or len(self._response_template.strip()) == 0
            ):
                raise ValueError(
                    "Response template is required for completions-only training."
                )
            if self._response_template.strip() != self._response_template:
                logger.warning(
                    f"Response template '{self._response_template}' contains "
                    "leading or trailing whitespaces. These will be ignored."
                )

                self._response_template = self._response_template.strip()

            if (
                self._instruction_template is None
                or len(self._instruction_template.strip()) == 0
            ):
                raise ValueError(
                    "Instruction template is required for completions-only training."
                )

            if self._instruction_template.strip() != self._instruction_template:
                logger.warning(
                    f"Instruction template '{self._instruction_template}' contains "
                    "leading or trailing whitespaces. These will be ignored."
                )

                self._instruction_template = self._instruction_template.strip()

            self.response_token_ids = self._tokenizer.encode(
                self._response_template, add_special_tokens=False
            )

            self.instruction_token_ids = self._tokenizer.encode(
                self._instruction_template, add_special_tokens=False
            )

            self._is_template_compatible_with_completions_only_training = False

    #
    # Batched Processing (for native HF .map())
    #
    @override
    def _transform_batch(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Optimized batched transform for SFT datasets.

        This method is called by HF `.map()` and provides batched tokenization
        for significant speedup (10-50x faster than serial processing).
        """
        if self._tokenizer is None:
            # No tokenizer - fall back to base implementation
            return super()._transform_batch(examples)

        if self._return_conversations:
            # Return conversations mode - fall back to base implementation
            return super()._transform_batch(examples)

        if self._assistant_only:
            # Assistant-only training requires special tokenization
            # Fall back to serial processing for now
            return super()._transform_batch(examples)

        # Get batch size from first column
        batch_size = len(next(iter(examples.values())))

        # Step 1: Convert all examples to Conversations
        conversations = []
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            conv = self.transform_conversation(example)
            conversations.append(conv)

        # Step 2: Apply chat template to all conversations (list of strings)
        texts = []
        for conv in conversations:
            text = self._tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=(self._task == "generation"),
            )
            texts.append(text)

        # Step 3: BATCH tokenize - this is where the major speedup comes from
        # One call to tokenizer with N texts vs N calls with 1 text
        tokenized = self._tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self._tokenizer.model_max_length,
            return_tensors=None,  # Return lists, not tensors
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
