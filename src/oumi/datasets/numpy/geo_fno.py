from pathlib import Path
from typing import Optional, Union

import pandas as pd
from typing_extensions import override

from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import register_dataset


@register_dataset("geo-fno/sigma_mesh_rr")
class GeoFNO(BaseMapDataset):
    default_dataset = "custom"

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initializes a new instance of the GeoFNO class."""
        if not dataset_path:
            raise ValueError("dataset_path must be provided")

        col_dict = {"sigma": "*_sigma_*.npy", "mesh": "*_XY_*.npy", "rr": "*_rr_*.npy"}

        super().__init__(
            dataset_name=dataset_name,
            dataset_path=str(dataset_path),
            column_to_filename=col_dict,
            **kwargs,
        )

        self._data = self._load_data()

    @override
    def transform(self, sample: pd.Series) -> dict:
        """Preprocesses the inputs in the given sample.

        Args:
            sample (dict): A dictionary containing the input data.

        Returns:
            dict: A dictionary containing the preprocessed input data.
        """
        return {"sigma": sample["sigma"], "mesh": sample["mesh"], "rr": sample["rr"]}
