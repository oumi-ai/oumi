# Test for a non-textual custom model (CNNClassifier).
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing_extensions import override

from oumi.builders import build_dataset, build_model
from oumi.core.configs import ModelParams, TrainingParams
from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import REGISTRY, RegistryType, register_dataset
from oumi.core.trainers.oumi_trainer import Trainer
from oumi.utils.logging import logger


@register_dataset("npz_file")
class NpzDataset(BaseMapDataset):
    """Loads dataset from Numpy .npz archive."""

    default_dataset = "custom"

    def __init__(
        self,
        *,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        npz_split_col: Optional[str] = None,
        npz_allow_pickle: bool = False,
        **kwargs,
    ) -> None:
        """Initializes a new instance of the NpzDataset class.

        Args:
            dataset_name: Dataset name.
            dataset_path: Path to .npz file.
            split: Dataset split.
            npz_split_col: Name of '.npz' array containing dataset split info.
                If unspecified, then the name "split" is assumed by default.
            npz_allow_pickle: Whether pickle is allowed when loading data
                from the npz archive.
            **kwargs: Additional arguments to pass to the parent class.
        Raises:
            ValueError: If dataset_path is not provided, or
                if .npz file contains data in unexpected format.
        """
        if not dataset_path:
            raise ValueError("`dataset_path` must be provided")
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=(str(dataset_path) if dataset_path is not None else None),
            split=split,
            **kwargs,
        )
        self._npz_allow_pickle = npz_allow_pickle
        self._npz_split_col = npz_split_col

        dataset_path = Path(dataset_path)
        if not dataset_path.is_file():
            raise ValueError(f"Path is not a file! '{dataset_path}'")
        elif dataset_path.suffix.lower() != ".npz":
            raise ValueError(f"File extension is not '.npz'! '{dataset_path}'")

        self._data = self._load_data()

    @staticmethod
    def _to_list(x: np.ndarray) -> list:
        # `pd.DataFrame` expects Python lists for columns
        # (elements can still be `ndarray`)
        if len(x.shape) > 1:
            return [x[i, ...] for i in range(x.shape[0])]
        return x.tolist()

    @override
    def _load_data(self) -> pd.DataFrame:
        data_dict = {}
        if not self.dataset_path:
            raise ValueError("dataset_path is empty!")
        with np.load(self.dataset_path, allow_pickle=self._npz_allow_pickle) as npzfile:
            feature_names = list(sorted(npzfile.files))
            if len(feature_names) == 0:
                raise ValueError(
                    f"'.npz' archive contains no data! '{self.dataset_path}'"
                )
            num_examples = None
            for feature_name in feature_names:
                col_data = npzfile[feature_name]
                assert isinstance(col_data, np.ndarray)
                if num_examples is None:
                    num_examples = col_data.shape[0]
                elif num_examples != col_data.shape[0]:
                    raise ValueError(
                        "Inconsistent number of examples for features "
                        f"'{feature_name}' and '{feature_names[0]}': "
                        f"{col_data.shape[0]} vs {num_examples}!"
                    )
                data_dict[feature_name] = self._to_list(col_data)

        logger.info(
            f"Loaded {num_examples} examples "
            f"with the features: {sorted(data_dict.keys())}!"
        )

        dataframe: pd.DataFrame = pd.DataFrame(data_dict)

        split_feature_name = (self._npz_split_col or "split") if self.split else None
        if split_feature_name:
            if split_feature_name not in dataframe:
                raise ValueError(
                    f"'.npz' doesn't contain data split info: '{split_feature_name}'!"
                )
            dataframe = pd.DataFrame(
                dataframe[dataframe[split_feature_name] == self.split].drop(
                    split_feature_name, axis=1
                ),
                copy=True,
            )
        return dataframe

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Preprocesses the inputs in the given sample."""
        return sample.to_dict()


def _convert_example_to_model_input(example: dict, device: torch.device) -> dict:
    return {
        key: (
            torch.from_numpy(value)
            if isinstance(value, np.ndarray)
            else torch.from_numpy(np.asarray(value))
        ).to(device, non_blocking=True)
        for key, value in example.items()
    }


def _get_dataset_length(dataset) -> int:
    result = 0
    for _ in dataset:
        result += 1
    return result


class MyDatasets(NamedTuple):
    train_dataset: Any
    validation_dataset: Any
    test_dataset: Any


def _create_test_data() -> MyDatasets:
    # Generate small test dataset.
    train_samples_per_digit = 8
    validation_samples_per_digit = 3
    test_samples_per_digit = 2
    total_samples_per_digit = (
        train_samples_per_digit + validation_samples_per_digit + test_samples_per_digit
    )

    images_list: list[np.ndarray] = []
    labels_list: list[int] = []
    splits_list: list[str] = []
    for digit in range(10):
        # Encode digits as intensity and add some noise.
        images_list.append(
            float(digit)
            * np.ones(
                shape=(total_samples_per_digit, 1, 28, 28),
                dtype=np.float32,
            )
        )
        labels_list.extend([digit] * total_samples_per_digit)
        splits_list.extend(["train"] * train_samples_per_digit)
        splits_list.extend(["validation"] * validation_samples_per_digit)
        splits_list.extend(["test"] * test_samples_per_digit)

    images = np.concatenate(images_list) / 11.0
    labels = np.stack(labels_list)
    splits = np.stack(splits_list)

    np.random.seed(42)
    images = images + np.random.random_sample(images.shape) * 0.05

    # Shuffle all arrays in unison.
    indices = np.random.permutation(len(labels))
    images = np.asarray(images[indices, ...], dtype=np.float32)  # cast as float32
    labels = labels[indices]
    splits = splits[indices]

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=tmp_dir
        ) as tmp_npz_file,
    ):
        npz_filename = tmp_npz_file.name
        np.savez_compressed(npz_filename, images=images, labels=labels, split=splits)

        dataset_cls = REGISTRY.get("npz_file", RegistryType.DATASET)
        assert dataset_cls is not None

        train_dataset = build_dataset(
            dataset_name="npz_file",
            tokenizer=None,
            split="train",
            use_torchdata=True,
            dataset_path=npz_filename,
        )
        assert _get_dataset_length(train_dataset) == 10 * train_samples_per_digit

        validation_dataset = build_dataset(
            dataset_name="npz_file",
            tokenizer=None,
            split="validation",
            use_torchdata=True,
            dataset_path=npz_filename,
        )
        assert (
            _get_dataset_length(validation_dataset) == 10 * validation_samples_per_digit
        )

        test_dataset = build_dataset(
            dataset_name="npz_file",
            tokenizer=None,
            split="test",
            use_torchdata=True,
            dataset_path=npz_filename,
        )
        assert _get_dataset_length(test_dataset) == 10 * test_samples_per_digit

        return MyDatasets(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )


def _validate_prediction(model: torch.nn.Module, dataset, *, max_iters: int = 5):
    with torch.no_grad():
        model_device = next(model.parameters()).device
        batch_size = 2

        for idx, test_input in enumerate(DataLoader(dataset, batch_size=batch_size)):
            if idx >= max_iters:
                break
            inputs = _convert_example_to_model_input(test_input, device=model_device)

            # Without label
            outputs = model(images=inputs["images"])
            assert outputs.keys() == ({"logits"})
            assert isinstance(outputs["logits"], torch.Tensor)
            logits = outputs["logits"].cpu().numpy()
            assert logits.shape == (batch_size, 10)
            assert logits.dtype == np.float32

            # With label
            outputs = model(images=inputs["images"], labels=inputs["labels"])
            assert outputs.keys() == ({"logits", "loss"})
            assert isinstance(outputs["logits"], torch.Tensor)
            logits = outputs["logits"].cpu().numpy()
            assert logits.shape == (batch_size, 10)
            assert logits.dtype == np.float32
            loss = outputs["loss"].cpu().numpy()
            assert loss.shape == ()
            assert loss.dtype == np.float32


def test_basic_training_and_prediction():
    test_data = _create_test_data()

    # Create a model.
    model_params = ModelParams(
        model_name="CnnClassifier",
        load_pretrained_weights=False,
        model_kwargs={
            "image_width": 28,
            "image_height": 28,
            "in_channels": 1,
            "output_dim": 10,
        },
    )
    model = build_model(model_params)

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
    ):
        training_params = TrainingParams(
            output_dir=str(Path(tmp_dir) / "cnn_classifier_test_output"),
            per_device_train_batch_size=32,
            per_device_eval_batch_size=8,
            eval_strategy="steps",
            eval_steps=50,
            max_steps=100,
            save_steps=0,
            logging_steps=50,
            optimizer="adam",
            learning_rate=2e-3,
        )

        trainer = Trainer(
            model=model,
            processing_class=None,  # No tokenizer! The custom model is non-textual
            args=training_params,
            train_dataset=test_data.train_dataset,
            dataloader_num_workers=2,
            dataloader_prefetch_factor=32,
        )

        trainer.train()

        _validate_prediction(trainer.model, test_data.test_dataset)
