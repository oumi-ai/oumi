(other-datasets)=
# Other Datasets

In addition to the common LLM dataset formats (e.g., [Pretraining](pretraining_datasets.md), [SFT](sft_datasets.md), [VL-SFT](vl_sft_datasets.md)),
Oumi infrastructure also allows users to define arbitrary ad-hoc datasets,
which can be used not just for text-centric LLM models, but for alternative model architectures
and applications such as Vision models (e.g., convolutional networks), scientific computing, etc.

This can be accomplished by defining a subclass of {py:class}`~oumi.core.datasets.BaseMapDataset` or {py:class}`~oumi.core.datasets.BaseIterableDataset`. A {py:class}`~oumi.core.datasets.BaseIterableDataset` is great for data streamed online, or for large datasets (e.g., hundreds of GBs) due to its lazy loading behavior, while {py:class}`~oumi.core.datasets.BaseMapDataset` should be the default choice for everything else (e.g., datasets that can be fully loaded into memory).

To give a concrete example, let's show how to add support for datasets stored in Numpy `.npz` file format:

(sample-custom-numpy-dataset)=
## NumPy Dataset

The popular `numpy` library defines `.npy` and `.npz` file formats [[details](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html)],
which can be used to [save](https://numpy.org/doc/stable/reference/generated/numpy.save.html) arbitrary multi-dimensional arrays ([`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)):

1. `.npy` file contains a single `np.ndarray`
2. `.npz` is an archive that contains a collection of multiple `np.ndarray`-s, with optional support for [data compression](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)


### Adding a New Numpy (.npz) Dataset

To add a new Oumi dataset that can load data from `.npz` files, follow these steps:

1. Subclass {py:class}`~oumi.core.datasets.BaseMapDataset`
2. Implement the methods to handle initialization, data loading, and data transformation.

Here's a basic example, which shows how to do that:

```python
from pathlib import Path
from typing import Optional, Union
from typing_extensions import override
import numpy as np
import pandas as pd

from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import register_dataset


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
            split: Dataset split. If unspecified then the whole dataset is loaded.
            npz_split_col: Name of '.npz' array containing dataset split info.
                If unspecified, then the name "split" is assumed by default.
            npz_allow_pickle: Whether pickle is allowed when loading data from the '.npz' archive.
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
        # `pd.DataFrame` expects Python lists for columns (elements can still be `ndarray`)
        if len(x.shape) > 1:
            return [x[i, ...] for i in range(x.shape[0])]
        return x.tolist()

    @override
    def _load_data(self) -> pd.DataFrame:
        data_dict: dict[str, np.ndarray] = {}
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

        dataframe = pd.DataFrame(data_dict)

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
        # Just return the data as a `dict`.
        return sample.to_dict()
```

```{note}
The `.npz` file format can be used to load images, vector fields, financial, medical/health data, and other new data types.
```

To use the custom dataset, add the following section to your {py:class}`~oumi.core.configs.TrainingConfig`:

```yaml
...
data:
  train:
    datasets:
      - dataset_name: "npz_file" # Custom dataset type defined above for .npz archives
        dataset_path: "/your_dir/mnist.npz" # File name of your `.npz` archive
        split: "train"
...
```

You can review the {gh}`➿ Training CNN on Custom Dataset <notebooks/Oumi - Training CNN on Custom Dataset.ipynb>` notebook for a complete example.
Additional information is available in [→ Custom Models](/resources/models/custom_models).

### Using Custom Datasets via the CLI

See {doc}`/user_guides/customization` to quickly enable your dataset when using the CLI.
