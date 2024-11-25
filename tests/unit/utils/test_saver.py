import tempfile

import pytest

from oumi.utils.saver import (
    load_infer_prob,
    load_infer_prob_csv,
    save_infer_prob,
    save_infer_prob_csv,
)

TEMP_FILENAME = "temp_filename"
PROBABILITIES = [
    [
        [0.01, 0.02, 0.03, 0.04],
        [0.11, 0.12, 0.13, 0.14],
    ],
    [
        [1.01, 1.02, 1.03, 1.04],
        [1.11, 1.12, 1.13, 1.14],
    ],
]


@pytest.mark.parametrize(
    "num_batches,batch_size,save_fn,load_fn",
    [
        (1, 1, save_infer_prob, load_infer_prob),
        (1, 2, save_infer_prob, load_infer_prob),
        (2, 1, save_infer_prob, load_infer_prob),
        (2, 2, save_infer_prob, load_infer_prob),
        (1, 1, save_infer_prob_csv, load_infer_prob_csv),
        (1, 2, save_infer_prob_csv, load_infer_prob_csv),
        (2, 1, save_infer_prob_csv, load_infer_prob_csv),
        (2, 2, save_infer_prob_csv, load_infer_prob_csv),
    ],
)
def test_save_load_infer_probs(num_batches, batch_size, save_fn, load_fn):
    probabilities = []
    for batch_no in range(num_batches):
        batch_probabilities = []
        for batch_index in range(batch_size):
            batch_probabilities.append(PROBABILITIES[batch_no][batch_index])
        probabilities.append(batch_probabilities)

    with tempfile.TemporaryDirectory() as output_temp_dir:
        # Save probabilities to a file.
        save_file = f"{output_temp_dir}/{TEMP_FILENAME}"
        save_fn(save_file, probabilities)

        # Load probabilities from file and compare to the original ones.
        loaded_probabilities = load_fn(save_file)
        assert probabilities == loaded_probabilities
