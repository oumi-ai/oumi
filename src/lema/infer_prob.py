from typing import List, Tuple

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
    acceptable_logits: List[str],
) -> List[List[List[float]]]:
    """Calculates the inference probabilities for the next logits to be generated.

    Args:
        model_params: The configuration object containing the model parameters.
        input: A list of text prompts of shape (num_batches, batch_size).
        acceptable_logits: The logits that are considered acceptable to be generated.
          The function will return the generation probabilities for each of these.

    Returns:
        object: A 2D list of of shape (num_batches, batch_size). Each item of the list
          is a list of probabilities (one probability per each acceptable logit).
    """
    tokenizer = build_tokenizer(model_params)
    model = build_model(model_params)
    model_device = next(model.parameters()).device

    # Tokenization of input (in place, batch mode).
    for batch_index, batch in enumerate(input):
        input[batch_index] = tokenizer(batch, return_tensors="pt", padding=True).to(
            model_device
        )

    # Tokenization of acceptable outputs (i.e. next logit to be generated).
    acceptable_logits_enc = tokenizer(
        acceptable_logits, add_special_tokens=False, padding=False, truncation=False
    )

    # Ensure each acceptable logit is encoded into a single token.
    for encoded_logit in acceptable_logits_enc.input_ids:
        if len(encoded_logit) != 1:
            raise ValueError("Not all `acceptable_logits` map to a single token.")

    # Flatten to a list of encoded tokens (corresponding to acceptable logits).
    acceptable_logits_enc = [tokens[0] for tokens in acceptable_logits_enc.input_ids]

    # Generate next logit probabilities (batch mode).
    output = []
    for batch_index in tqdm(range(len(input)), desc="Generating Logit Probs"):
        logit_probs = model(input[batch_index].input_ids)  # type: ignore
        next_logit_probabilities = logit_probs.logits[:, -1, :]  # -1 for next logit.
        output.append(next_logit_probabilities)

    # Reduce next logit probabilities to only the acceptable next logits.
    inference_probs = []
    for batch in output:
        inference_probs_batch = []
        for next_logit_probs in batch:
            acceptable_logit_probs = []
            for acceptable_logit in acceptable_logits_enc:
                acceptable_logit_prob = next_logit_probs[acceptable_logit]
                with torch.no_grad():
                    acceptable_logit_probs.append(
                        acceptable_logit_prob.cpu().numpy().item()
                    )
            inference_probs_batch.append(softmax(acceptable_logit_probs).tolist())
        inference_probs.append(inference_probs_batch)

    return inference_probs
