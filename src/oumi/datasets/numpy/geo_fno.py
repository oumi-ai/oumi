from pathlib import Path
from typing import Optional, Union

from oumi.core.datasets import BaseMapDataset
from oumi.core.registry import register_dataset


@register_dataset("geo-fno/sigma_mesh_rr")
class GeoFNO(BaseMapDataset):
    default_dataset = "custom"

    def __init__(
        self,
        dataset_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Initializes a new instance of the GeoFNO class."""
        if dataset_path is not None:
            raise ValueError("dataset_path must be provided")

        col_dict = {"sigma": "*_sigma_*.npy", "mesh": "*_XY_*.npy", "rr": "*_rr_*.npy"}

        super().__init__(
            dataset_path=dataset_path, column_to_filename=col_dict, **kwargs
        )
