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

import queue
import random
import threading
from typing import Callable, Optional

import datasets
import torch
from torch.utils.data import IterableDataset

from oumi.core.tokenizers import BaseTokenizer
from oumi.utils.logging import logger

_LARGEST_PRIORITY_VALUE = 2**20
_SMALLEST_PRIORITY_VALUE = 0
_END_PRIORITY_VALUE = _LARGEST_PRIORITY_VALUE + 1


class PretrainingAsyncTextDataset(IterableDataset):
    """Iterable dataset that returns constant length chunks of tokens.

    Prefetches, formats, and tokenizes asynchronously from main thread.

    Based on TRL's ConstantLengthDataset class.
    """

    def __init__(
        self,
        tokenizer: Optional[BaseTokenizer],
        dataset: datasets.Dataset,
        dataset_text_field: Optional[str] = None,
        formatting_func: Optional[Callable] = None,
        infinite: bool = False,
        seq_length: int = 1024,
        sequence_buffer_size: int = 1024,
        eos_token_id: int = 0,
        shuffle: bool = False,
        append_concat_token: bool = True,
        add_special_tokens: bool = True,
        pretokenized: bool = True,
    ):
        """Iterable dataset that returns constant length chunks of tokens.

        Args:
            tokenizer (`BaseTokenizer`):
                The tokenizer used for converting strings to tokens.
            dataset (`dataset.Dataset`):
                Dataset of text samples.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text.
                Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization.
                Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question} ### Answer: {answer}"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
                Should set to global_batch_size * 2 for minimum delay.
            sequence_buffer_size (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in
                text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have
                an EOS token.
            shuffle (`bool`, *optional*, defaults to False):
                Shuffle the examples before they are returned.
            append_concat_token (`bool`, *optional*, defaults to True):
                If true, appends `eos_token_id` at the end of each sample being packed.
            add_special_tokens (`bool`, *optional*, defaults to True):
                If true, tokenizers adds special tokens to each sample being packed.
            pretokenized (`bool`, *optional*, defaults to False):
                If true, the dataset is already tokenized and formatted, and each sample
                is expected to have an "input_ids" field.
        """
        self.tokenizer = tokenizer

        if not pretokenized and tokenizer is None:
            raise ValueError("Tokenizer is required when dataset is not pretokenized.")

        if tokenizer is None or tokenizer.eos_token_id is None:
            logger.warning(
                "The passed tokenizer does not have an EOS token. We will use the"
                " passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure "
                "to pass the correct eos_token_id."
            )

        self.concat_token_id = (
            tokenizer.eos_token_id
            if tokenizer is not None and tokenizer.eos_token_id
            else eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        self.shuffle = shuffle
        self.pretokenized = pretokenized

        if shuffle:
            self.tokenized_example_queue = queue.PriorityQueue(
                maxsize=sequence_buffer_size
            )
        else:
            self.tokenized_example_queue = queue.Queue(maxsize=sequence_buffer_size)

        if formatting_func is not None:
            self.formatting_func = formatting_func

            if formatting_func.__code__.co_argcount != 1:
                logger.warning(
                    "The passed formatting_func does not have exactly 1 argument. Note "
                    "that additional arguments will remain unused."
                )
        elif dataset_text_field is not None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = lambda x: x

    @property
    def column_names(self) -> list[str]:
        """Returns the column names of the dataset."""
        return ["input_ids", "labels"]

    def _add_example_to_queue(self, example):
        """Adds a single example to the queue."""
        # Shuffle by using a priority queue with random priority values
        # Note that the tensors themselves are identical,
        # Only the order they are returned is shuffled.
        priority = _SMALLEST_PRIORITY_VALUE
        if self.shuffle:
            priority = random.randint(_SMALLEST_PRIORITY_VALUE, _LARGEST_PRIORITY_VALUE)

        self.tokenized_example_queue.put(
            (
                priority,
                {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                },
            )
        )

    def _dataset_iterator_worker(self):
        iterator = iter(self.dataset)
        token_buffer = []
        while True:
            token_count = len(token_buffer)
            try:
                next_sample = next(iterator)
            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                    logger.warning(
                        "Reached the end of the dataset, resetting to the start."
                    )

                    continue
                else:
                    break

            if not self.pretokenized:
                formatted_input = self.formatting_func(next_sample)
                if self.tokenizer is not None:
                    tokenized = self.tokenizer(
                        [formatted_input],
                        add_special_tokens=self.add_special_tokens,
                        truncation=False,
                    )
                else:
                    raise ValueError("Tokenizer is not initialized")
                tokenized_input = tokenized["input_ids"][0]  # type: ignore - Returns Sequence[EncodingFast]
            else:
                if "input_ids" not in next_sample:
                    raise ValueError(
                        "The dataset is pretokenized but does not have an 'input_ids' "
                        "field."
                    )
                tokenized_input = next_sample["input_ids"]  # type: ignore - Returns Sequence[EncodingFast]

            if self.append_concat_token:
                tokenized_input = tokenized_input + [self.concat_token_id]

            token_count += len(tokenized_input)
            token_buffer.extend(tokenized_input)

            # Not enough tokens to make an example, continue.
            if token_count < self.seq_length:
                continue

            examples = []
            last_index = -1
            for i in range(0, len(token_buffer), self.seq_length):
                input_ids = token_buffer[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
                    last_index = i + self.seq_length
            token_buffer = token_buffer[last_index:]

            for example in examples:
                self._add_example_to_queue(example)

        # Add any remaining tokens as a final example that's padded
        token_limit = 0
        if self.append_concat_token:
            # Set limit to 1 to account for trailing concat token
            token_limit = 1

        num_remaining_tokens = len(token_buffer)
        if num_remaining_tokens > token_limit:
            trailing_example = token_buffer + [
                self.concat_token_id
                for _ in range(self.seq_length - num_remaining_tokens)
            ]
            self._add_example_to_queue(trailing_example)

        # Signal that there are no more samples, have this be the last value
        self.tokenized_example_queue.put((_END_PRIORITY_VALUE, None))

    def __iter__(self):
        """Iterates through the dataset with most work on a separate thread."""
        # Set worker thread to daemon so it dies when the program finishes.
        worker_thread = threading.Thread(
            target=self._dataset_iterator_worker, args=(), daemon=True
        )
        worker_thread.start()
        while True:
            _, tensors = self.tokenized_example_queue.get()
            if tensors is None:
                break
            yield tensors

        worker_thread.join()
