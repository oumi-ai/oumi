from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.types import ModelParams


def softmax(x, axis=None):
    """Computes the softmax function.

    The softmax function transforms each element of a collection by computing the
    exponential of each element divided by the sum of the exponentials of all the
    elements.

    Note: This implementation is from scipy. We should consider replacing it with a
    call to scipy.special.softmax(), if we add the scipy dependency for other
    functionalities in the future.
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)


def most_probable_logits(
    tokenizer: PreTrainedTokenizerBase, logit_probs: List[float], count: int = 3
) -> List[Tuple[str, float]]:
    """Return the `count` most probable next logits, with their probabilities."""
    indices = np.argsort(logit_probs)
    indices = indices[::-1][:count]  # Reverse and only keep `count` items.
    return [(tokenizer.decode(index), logit_probs[index]) for index in indices]


def infer_prob(
    model_params: ModelParams,
    input: List[List[str]],
    acceptable_logits: Optional[List[str]] = None,
) -> List[List[List[float]]]:
    """Calculates the inference probabilities for the next logits to be generated.

    Args:
        model_params: The configuration object containing the model parameters.
        input: A list of text prompts of shape (num_batches, batch_size).
        acceptable_logits: The logits that are considered acceptable to be generated.
          The function will return the generation probabilities for each of these. If
          not provided (= None), the probabilities for the entire tokenizer's vocabulary
          will be returned.

    Returns:
        object: A 2D list of shape (num_batches, batch_size). Each item is another list
          of the probabilities (one probability for very acceptable logit).
    """
    tokenizer = build_tokenizer(model_params)
    logits_vocab = set(tokenizer.get_vocab())
    logits_enc_vocab = set(tokenizer.get_vocab().values())

    model = build_model(model_params)
    model_device = next(model.parameters()).device

    # Tokenization of input (in place, batch mode).
    for batch_index, batch in enumerate(input):
        input[batch_index] = tokenizer(batch, return_tensors="pt", padding=True).to(
            model_device
        )

    # Ensure the `acceptable_logits` are valid.
    if not acceptable_logits:
        # If no list of acceptable logits provided, use the entire vocabulary.
        acceptable_logits = list(logits_vocab)
    else:
        # If provided with a list of logits, ensure these exist in the vocabulary.
        for logit in acceptable_logits:
            if logit not in logits_vocab:
                raise ValueError(f"Logit `{logit}` NOT found in vocabulary")

    # Tokenization of acceptable outputs (i.e. next logit to be generated).
    acceptable_logits_enc = tokenizer.convert_tokens_to_ids(acceptable_logits)
    for logit_enc in acceptable_logits_enc:
        if logit_enc not in logits_enc_vocab:
            # For sanity checking, we also need to ensure that the encoded logits exist
            # in the tokenizer's vocabulary. This check will fail primarily due to bugs
            # in custom tokenizer implementations, or incompatible tokenizer types.
            raise ValueError(f"Enc logit `{logit_enc}` NOT found in vocabulary")
        if logit_enc >= len(logits_vocab):
            # The `logit_enc` will be utimately used as an index, to extract the
            # probability of the logit, from a vocabulary-sized tensor. So, it must NOT
            # be larger than the vocabulary size under any circumstances.
            raise ValueError(f"Enc logit `{logit_enc}` larger than vocabulary size")

    # Generate next logit probabilities (batch mode).
    # Explanation:
    #     Gets the next logit probabilities, i.e. `logit probs.logits`; this is a tensor
    #     of shape [batch_size, num_input_logits, vocabulary_size].
    #     - batch_size: The output is batched, since our input (`input`) is batched.
    #     - num_input_logits: The probability of the next logit, for each logit that is
    #       included in our input prompt. We are only interested in the next logit that
    #       comes after the last logit of the input sequence, thus we will flatten this
    #       dimension and only look at the final (-1) logit probabilities.
    #     - vocabulary_size: We are provided with the probability for each possible
    #       logit that exists in the tokenizer's vocabulary, thus this dimension equals
    #       the size of the vocabulary.
    #     The `output` will be a 3D list [num_batches, batch_size, vocabulary_size].
    output = []
    for batch_index in tqdm(range(len(input)), desc="Generating Logit Probs"):
        with torch.no_grad():
            logit_probs = model(input[batch_index].input_ids)  # type: ignore
            final_logit_probs = logit_probs.logits[:, -1, :].tolist()

            # For most tokenizers, the model returns as many probabilities as the number
            # of logits that exist in the vocabulary. But, some models may return
            # more, and also include special tokens (such as "end of generation"), which
            # are not included in the vocabulary provided to the user.
            assert len(final_logit_probs[-1]) >= len(logits_vocab)
            output.append(final_logit_probs)

    def reduce_to_acceptable_probs(logit_probs: List[float]) -> List[float]:
        """Reduces the list of all logit probabilities to only the logits of interest.

        Takes as input the list of probabilities that correspond to all logits in the
        vocabulary and returns the list of probabilities for only the logits that the
        user explicitly requested (`acceptable_logits_enc`). Then, applies softmax,
        so that the new set of probabilities sums up to 1.
        """
        logit_probs = [logit_probs[logit] for logit in acceptable_logits_enc]
        return softmax(logit_probs).tolist()

    return [list(map(reduce_to_acceptable_probs, batch)) for batch in output]
