from pathlib import Path
from typing import Union

from oumi.utils.logging import logger


def is_cached_to_disk_hf_dataset(dataset_folder: Union[str, Path]) -> bool:
    """Detects whether a dataset was saved using `dataset.save_to_disk()`.

    Such datasets should be loaded using `datasets.Daataset.load_from_disk()`

    Returns:
        Whether the dataset was saved using `dataset.save_to_disk()` method.
    """
    if not dataset_folder:
        return False

    dataset_path: Path = Path(dataset_folder)

    if dataset_path.exists() and dataset_path.is_dir():
        for file_name in ("dataset_info.json", "state.json"):
            file_path: Path = dataset_path / file_name
            if not (file_path.exists() and file_path.is_file()):
                logger.warning(
                    f"The dataset {str(dataset_path)} is missing "
                    f"a required file: {file_name}."
                )
                return False
        return True

    return False
