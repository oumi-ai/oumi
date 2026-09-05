from unittest.mock import patch

import pandas as pd

from oumi.datasets.grpo.rar_dataset import RaRMedicineVerlGrpoDataset


def test_rar_medicine_verl_transform():
    raw_data = pd.DataFrame(
        [
            {
                "question": "  What is the diagnosis?  ",
                "reference_answer": "  Influenza  ",
            }
        ]
    )
    with patch.object(
        RaRMedicineVerlGrpoDataset,
        "_load_data",
        return_value=raw_data,
    ):
        dataset = RaRMedicineVerlGrpoDataset(split="train")

    row = dataset[0]

    assert row["prompt"][-1] == {
        "role": "user",
        "content": "What is the diagnosis?",
    }
    assert row["data_source"] == "anisha2102/RaR-Medicine"
    assert row["ability"] == "medicine"
    assert row["reward_model"] == {
        "style": "rule",
        "ground_truth": "Influenza",
    }
    assert row["extra_info"] == {
        "split": "train",
        "question": "What is the diagnosis?",
    }
