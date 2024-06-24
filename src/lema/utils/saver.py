import csv
from typing import List

import pandas as pd
import transformers

from lema.core.types import TrainingConfig
from lema.logging import logger


def save_model(config: TrainingConfig, trainer: transformers.Trainer) -> None:
    """Saves the model's state dictionary to the specified output directory.

    Args:
        config (TrainingConfig): The LeMa training config.
        trainer (transformers.Trainer): The trainer object used for training the model.

    Returns:
        None
    """
    output_dir = config.training.output_dir

    if config.training.use_peft:
        state_dict = {k: t for k, t in trainer.model.named_parameters() if "lora_" in k}
    else:
        state_dict = trainer.model.state_dict()

    trainer._save(output_dir, state_dict=state_dict)
    logger.info(f"Model has been saved at {output_dir}.")


def save_infer_prob(output_filepath: str, probabilities: List[List[List[float]]]):
    """Save batched probabilities into a parquet file."""
    df_probs = pd.DataFrame(probabilities)
    df_probs.to_parquet(output_filepath)


def load_infer_prob(
    input_filepath: str, num_labels: int = 0
) -> List[List[List[float]]]:
    """Retrieve batched probabilities from a parquet file."""

    def to_list(probs):
        """Ensure number of probabilities is the same for all entries."""
        probs_list = list(probs)
        nonlocal num_labels
        num_labels = num_labels or len(probs_list)
        if num_labels != len(probs_list):
            raise ValueError(
                f"Reading {input_filepath}: inconsistent number of probs"
                f" across entries: len({probs_list}) != {num_labels}"
            )
        return probs_list

    df_probs = pd.read_parquet(input_filepath)
    probabilities = df_probs.values.tolist()
    probabilities = [[to_list(probs) for probs in batch] for batch in probabilities]
    return probabilities


#  The inference probabilities (`probabilities`) are structured as follows:
#  (the example below assumes 4 batches of batch_size=2 and, for each of these,
#   4 probabilities corresponding to the multiple choices A, B, C, D)
#
#  [
#    [                                           <-- batch no 0:
#      [p_0_0_A, p_0_0_B, p_0_0_C, p_0_0_D],     <-- batch index = 0
#      [p_0_1_A, p_0_1_B, p_0_1_C, p_0_1_D],     <-- batch index = 1
#    ],
#    [                                           <-- batch no 1:
#      [p_1_0_A, p_1_0_B, p_1_0_C, p_1_0_D],     <-- batch index = 0
#      [p_1_1_A, p_1_1_B, p_1_1_C, p_1_1_D],     <-- batch index = 1
#    ],
#    [                                           <-- batch no 2:
#      [p_2_0_A, p_2_0_B, p_2_0_C, p_2_0_D],     <-- batch index = 0
#      [p_2_1_A, p_2_1_B, p_2_1_C, p_2_1_D],     <-- batch index = 1
#    ],
#    [                                           <-- batch no 3:
#      [p_3_0_A, p_3_0_B, p_3_0_C, p_3_0_D],     <-- batch index = 0
#      [p_3_1_A, p_3_1_B, p_3_1_C, p_3_1_D],     <-- batch index = 1
#    ]
#  ]
#
#  We save these into a .csv file of the following format:
#  - Every row corresponds to a batch.
#  - Within each row, the batch items are strings separated by comma (,).
#  - Each item (string) contains a list of probabilities (floats).
#
#              batch index = 0                        batch index = 1           batch no
#   <-------------------------------->  ,  <-------------------------------->       |
# "[p_0_0_A, p_0_0_B, p_0_0_C, p_0_0_D]","[p_0_1_A, p_0_1_B, p_0_1_C, p_0_1_D]"  <--0
# "[p_1_0_A, p_1_0_B, p_1_0_C, p_1_0_D]","[p_1_1_A, p_1_1_B, p_1_1_C, p_1_1_D]"  <--1
# "[p_2_0_A, p_2_0_B, p_2_0_C, p_2_0_D]","[p_2_1_A, p_2_1_B, p_2_1_C, p_2_1_D]"  <--2
# "[p_3_0_A, p_3_0_B, p_3_0_C, p_3_0_D]","[p_3_1_A, p_3_1_B, p_3_1_C, p_3_1_D]"  <--3
#


def save_infer_prob_csv(output_filepath: str, probabilities: List[List[List[float]]]):
    """Save batched probabilities into a csv file."""
    with open(output_filepath, "w") as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerows(probabilities)


def load_infer_prob_csv(
    input_filepath: str, num_labels: int = 0
) -> List[List[List[float]]]:
    """Retrieve batched probabilities from a csv file."""
    try:
        with open(input_filepath, "r") as read_obj:
            csv_reader = csv.reader(read_obj)

            probabilities = []
            for batch in csv_reader:
                probabilities_batch = []
                for entry in batch:
                    probs_list = str_to_float_list(entry)

                    # Number of probabilities must be the same for all entries.
                    num_labels = num_labels or len(probs_list)
                    if num_labels != len(probs_list):
                        raise ValueError(
                            f"Reading {input_filepath}: inconsistent number of probs"
                            f" across entries: len({probs_list}) != {num_labels}"
                        )

                    probabilities_batch.append(probs_list)
                probabilities.append(probabilities_batch)
            return probabilities
    except FileNotFoundError:
        raise FileNotFoundError(f"{load_infer_prob}: Path {input_filepath} not found!")


def str_to_float_list(input: str) -> List[float]:
    """Convert an `str` representing a list of `floats` to an actual list of `floats`.

    Example: input: `[1.1, 2.2, 3.3]` => output: [1.1, 2.2, 3.3]
    """
    # 1) Get rid of '[' and ']'.
    if (input[0] != "[") or (input[-1] != "]"):
        raise ValueError(
            f"Input `{input}` must start with '[' and end with ']' to represent a list"
        )
    input = input[1:-1]

    # 2) Convert string to a list of items.
    list_of_items = input.split(", ")
    if not len(list_of_items):
        raise ValueError(f"List `{list_of_items}` does NOT contain any items")

    # 3) Cast all list items to `float`.
    try:
        list_of_floats = [float(item) for item in list_of_items]
    except ValueError:
        raise ValueError(f"List `{list_of_items}` should contain probabilities")

    return list_of_floats
