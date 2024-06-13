from typing import List

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from lema.builders import (
    build_model,
    build_tokenizer,
)
from lema.core.types import ModelParams


def softmax(x: List[float]) -> List[float]:
    """Compute softmax values for each sets of scores in x."""
    return (np.exp(x) / np.sum(np.exp(x), axis=0)).tolist()


def most_probable_logits(
    tokenizer: PreTrainedTokenizerBase, logit_probs: List[float], count: int = 3
):
    """Return the `count` most probable next logits, with their probabilities."""
    logit_probs_sorted = sorted(set(logit_probs), reverse=True)
    probable_logit_indices = []
    probable_logits = []
    for probability in logit_probs_sorted:
        indices = [i for i, p in enumerate(logit_probs) if p == probability]
        probable_logit_indices.extend(indices)
        if len(probable_logit_indices) >= count:
            break
    probable_logit_indices = probable_logit_indices[:count]

    for index in probable_logit_indices:
        probable_logits.append((tokenizer.decode(index), logit_probs_sorted[index]))
    return probable_logits


def infer_prob(
    model_params: ModelParams,
    input: List[List[str]],
    acceptable_logits: List[str],
) -> List[List[List[float]]]:
    """Calculate the inference probabilities for the next logits to be generated.

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
        batch_tokenized = tokenizer(batch, return_tensors="pt", padding=True)
        batch_tokenized = batch_tokenized.to(model_device)
        input[batch_index] = batch_tokenized

    # Tokenization of acceptable outputs (i.e. next logit to be generated).
    acceptable_logits_enc = tokenizer(
        acceptable_logits, add_special_tokens=False, padding=False, truncation=False
    )

    # Ensure each acceptable logit is encoded into a single token.
    assert all([len(tokens) == 1 for tokens in acceptable_logits_enc.input_ids])

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
                acceptable_logit_probs.append(
                    acceptable_logit_prob.detach().cpu().numpy().item()
                )
            inference_probs_batch.append(softmax(acceptable_logit_probs))
        inference_probs.append(inference_probs_batch)

    return inference_probs
