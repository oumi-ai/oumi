from typing import List, Optional, Tuple, cast

import numpy as np
import torch
from tqdm import tqdm

from lema.builders import build_model, build_tokenizer
from lema.core.types import ModelParams
from lema.core.types.base_tokenizer import BaseTokenizer
from lema.utils.logging import logger
from lema.utils.saver import load_infer_prob, save_infer_prob


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


def most_probable_tokens(
    tokenizer: BaseTokenizer, token_probs: List[float], count: int
) -> List[Tuple[str, float]]:
    """Return the `count` most probable next tokens, with their probabilities."""
    indices = np.argsort(token_probs)
    indices = indices[::-1][:count]  # Reverse and only keep `count` items.
    return [(tokenizer.decode(index), token_probs[index]) for index in indices]


def infer_prob(
    model_params: ModelParams,
    input: List[List[str]],
    acceptable_tokens: Optional[List[str]] = None,
    input_filepath: Optional[str] = None,
    output_filepath: Optional[str] = None,
    enable_dp: bool = False,
) -> List[List[List[float]]]:
    """Calculates the inference probabilities for the next tokens to be generated.

    Args:
        model_params: The configuration object containing the model parameters.
        input: A list of text prompts of shape (num_batches, batch_size).
        acceptable_tokens: The tokens that are considered acceptable to be generated.
          The function will return the generation probabilities for each of these. If
          not provided (= None), the probabilities for the entire tokenizer's vocabulary
          will be returned.
        input_filepath: File path to read the inference probabilities from. If provided,
          this function will directly return these.
        output_filepath: File path to save the inference probabilities, after being
          computed, for future reference.
        enable_dp: Enable DataParallel (DP) execution if multiple GPUs are available.

    Returns:
        object: A 2D list of shape (num_batches, batch_size). Each item of the 2D list
        is another list of the probabilities (one probability per acceptable token).
    """
    if input_filepath:
        return load_infer_prob(input_filepath)

    tokenizer = build_tokenizer(model_params)
    token_vocab = set(tokenizer.get_vocab())
    token_id_vocab = set(tokenizer.get_vocab().values())

    if enable_dp:
        if not torch.cuda.is_available():
            raise ValueError("DataParallel (DP) execution requested but no GPUs found.")

        logger.info(
            "DataParallel (DP) execution enabled. "
            "Overriding device_map to default device."
        )
        model_params.device_map = "cuda"

    model = build_model(model_params, enable_dp=enable_dp)

    if enable_dp:
        # In DP, inputs should be on the default device
        data_device = model_params.device_map
    else:
        # inputs should be in the same device as the model
        data_device = next(model.parameters()).device

    # Tokenization of input (batch mode).
    # `input_tok` is a 2D list of tokenized prompts of shape (num_batches, batch_size).
    # Each tokenized prompt itself is a class `tokenizers.Encoding`. If the tokenizer is
    # a pure python tokenizer (i.e., not “Fast”), this class behaves just like a python
    # dictionary, which holds the tokenized prompts under the key `input_ids`.
    input_tok = []
    for batch in input:
        input_tok.append(tokenizer(batch, return_tensors="pt", padding=True))

    # Ensure the `acceptable_tokens` are valid.
    if not acceptable_tokens:
        # If no list of acceptable tokens provided, use the entire vocabulary.
        acceptable_tokens = list(token_vocab)
    else:
        # If provided with a list of tokens, ensure these exist in the vocabulary.
        for token in acceptable_tokens:
            if token not in token_vocab:
                raise ValueError(f"Token `{token}` NOT found in vocabulary")

    # Tokenization of acceptable outputs (i.e. next token to be generated).
    acceptable_token_ids = cast(
        List[int], tokenizer.convert_tokens_to_ids(acceptable_tokens)
    )
    for token_id in acceptable_token_ids:
        if token_id not in token_id_vocab:
            # For sanity checking, we need to ensure that the encoded tokens (ids) exist
            # in the tokenizer's vocabulary. This check may fail primarily due to bugs
            # in custom tokenizer implementations, or incompatible tokenizer types.
            raise ValueError(f"Token id `{token_id}` NOT found in vocabulary")
        if token_id >= len(token_id_vocab):
            # The `token_id` will be utimately used as an index, to extract the
            # probability of the token, from a vocabulary-sized tensor. So, it must NOT
            # be larger than the vocabulary size under any circumstances.
            raise ValueError(f"Token id `{token_id}` larger than vocabulary size")

    # Generate next token probabilities (batch mode).
    # Explanation:
    #     Gets next tokens' unnormalized probabilities (logits): `token probs.logits`.
    #     This is a tensor of shape [batch_size, num_input_tokens, vocabulary_size].
    #     - batch_size: The output is batched, since our input (`input_tok`) is batched.
    #     - num_input_tokens: The probability of the next token, for each token that is
    #       included in our input prompt. We are only interested in the next token that
    #       comes after the last token of our input sequence, thus we will flatten this
    #       dimension and only look at the final (-1) token probabilities.
    #     - vocabulary_size: We are provided with the generation probability for each
    #       possible token that exists in the tokenizer's vocabulary, thus this
    #       dimension equals the size of the vocabulary.
    #     The `output` will be a 3D list [num_batches, batch_size, vocabulary_size].
    output = []
    for batch_index in tqdm(range(len(input_tok)), desc="Generating Token Logits"):
        with torch.no_grad():
            inputs = input_tok[batch_index].input_ids.to(data_device)
            token_logits = model(inputs)  # type: ignore
            token_logits = token_logits.logits[:, -1, :].tolist()

            # For most tokenizers, the model returns as many probabilities as the number
            # of tokens that exist in the vocabulary. But, some models may return
            # more, and also include special tokens (such as "end of generation"), which
            # are not included in the vocabulary provided to the user. Thus the ">=".
            assert len(token_logits[-1]) >= len(token_vocab)
            output.append(token_logits)

    def reduce_to_acceptable_tokens(token_logits: List[float]) -> List[float]:
        """Reduces the list of token logits to only the token logits of interest.

        Takes as input the list of logits that correspond to all tokens in the
        vocabulary and returns the list of logits for only the tokens that the
        user explicitly requested (`acceptable_token_ids`). Then, applies softmax,
        so that the corresponding probabilities are normalized to sum up to 1.
        """
        token_logits = [token_logits[token_id] for token_id in acceptable_token_ids]
        return softmax(token_logits).tolist()

    probabilities = [list(map(reduce_to_acceptable_tokens, batch)) for batch in output]
    if output_filepath:
        save_infer_prob(output_filepath, probabilities)

    return probabilities
