import ast
import csv
from typing import List

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


def save_infer_prob(write_file: str, probabilities: List[List[List[float]]]):
    """Save batched probabilities into a csv file."""
    with open(write_file, "w") as write_obj:
        csv_writer = csv.writer(write_obj)
        csv_writer.writerows(probabilities)


def load_infer_prob(read_file: str, num_labels: int = 0) -> List[List[List[float]]]:
    """Retrieve batched probabilities from a csv file."""
    try:
        with open(read_file, "r") as read_obj:
            csv_reader = csv.reader(read_obj)

            probabilities = []
            for batch in csv_reader:
                probabilities_batch = []
                for entry in batch:
                    probs_list = ast.literal_eval(entry)

                    # Sanity check: probabilities must be in a list.
                    if not isinstance(probs_list, list):
                        raise ValueError(
                            f"Reading {read_file}: probabilities must be contained in "
                            f"lists, but instead found {probs_list}."
                        )

                    # Sanity check: probabilities must be of type `float``.
                    if not all(isinstance(p, float) for p in probs_list):
                        raise ValueError(
                            f"Reading {read_file}: list items should be of type `float`"
                            f" (probabilities), but instead found {probs_list}."
                        )

                    # Sanity check: probability counts must be the same for all entries.
                    num_labels = num_labels or len(probs_list)
                    if num_labels != len(probs_list):
                        raise ValueError(
                            f"Reading {read_file}: inconsistent number of probabilities"
                            f" across entries: len({probs_list}) != {num_labels}"
                        )

                    probabilities_batch.append(probs_list)
                probabilities.append(probabilities_batch)
            return probabilities
    except FileNotFoundError:
        raise FileNotFoundError(f"{load_infer_prob}: File {read_file} not found!")
