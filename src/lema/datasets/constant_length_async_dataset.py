import queue
import random
import threading

import torch
from torch.utils.data import IterableDataset

from lema.logging import logger

_LARGEST_PRIORITY_VALUE = 2**20
_SMALLEST_PRIORITY_VALUE = 0
_END_PRIORITY_VALUE = _LARGEST_PRIORITY_VALUE + 1


class ConstantLengthAsyncDataset(IterableDataset):
    """Iterable dataset that returns constant length chunks of tokens.

    Prefetches, formats, and tokenizes asynchronously from main thread.

    Based on TRL's ConstantLengthDataset class.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        eos_token_id=0,
        shuffle=False,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        """Iterable dataset that returns constant length chunks of tokens.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
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
            num_of_sequences (`int`, *optional*, defaults to `1024`):
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
        """
        self.tokenizer = tokenizer

        if tokenizer.eos_token_id is None:
            logger.warn(
                "The passed tokenizer does not have an EOS token. We will use the"
                " passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure "
                "to pass the correct eos_token_id."
            )

        self.concat_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.append_concat_token = append_concat_token
        self.add_special_tokens = add_special_tokens
        self.shuffle = shuffle

        if shuffle:
            self.tokenized_example_queue = queue.PriorityQueue(maxsize=num_of_sequences)
        else:
            self.tokenized_example_queue = queue.Queue(maxsize=num_of_sequences)

        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            if formatting_func.__code__.co_argcount > 1:
                logger.warn(
                    "The passed formatting_func has more than one argument. Usually "
                    "that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of "
                    "the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        """Get length of underlying dataset."""
        return len(self.dataset)

    def _add_example_to_queue(self, example):
        """Add a single example to the queue."""
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
        # TODO: Increase to more than 1 thread
        iterator = iter(self.dataset)
        more_examples = True
        token_buffer = []
        while more_examples:
            token_count = len(token_buffer)
            try:
                formatted_input = self.formatting_func(next(iterator))
            except StopIteration:
                if self.infinite:
                    iterator = iter(self.dataset)
                    logger.warn(
                        "The dataset reached end, iterator is reset to the start."
                    )
                else:
                    more_examples = False
                    break

            tokenized_input = self.tokenizer(
                [formatted_input],
                add_special_tokens=self.add_special_tokens,
                truncation=False,
            )["input_ids"][0]

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
        """Iterate through the dataset with most work on a separate thread."""
        # Set worker thread to daemon so it dies when the program finishes.
        worker_thread = threading.Thread(
            target=self._dataset_iterator_worker, daemon=True
        )
        worker_thread.start()
        while True:
            _, tensors = self.tokenized_example_queue.get()
            if tensors is None:
                break
            self.current_size += 1
            yield tensors

        worker_thread.join()
