from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from oumi.utils.logging import logger


def _try_resolve_column_numpy_files(
    dataset_folder: Union[str, Path],
    column_to_filename_dict: dict[str, str],
    *,
    strict: bool = False,
) -> Optional[dict[str, Path]]:
    if not dataset_folder or len(column_to_filename_dict) == 0:
        if strict:
            raise ValueError(
                "Empty "
                + (
                    "dataset_folder"
                    if not dataset_folder
                    else "column_to_filename_dict"
                )
                + "!"
            )
        return None

    dataset_dir: Path = Path(dataset_folder)
    if not (dataset_dir.exists() and dataset_dir.is_dir()):
        if strict:
            raise ValueError(
                "dataset_folder doesn't exist or not a directory! "
                f"Path: {dataset_dir}"
            )
        return None

    result: dict[str, Path] = {}
    for feature_name, pattern in column_to_filename_dict.items():
        matched_files: list[Path] = list(sorted(dataset_dir.glob(pattern)))
        if len(matched_files) > 1:
            raise ValueError(
                f"Multiple files ({len(matched_files)}) "
                f"found for the feature '{feature_name}'. "
                f"Pattern: {pattern}"
            )
        elif len(matched_files) == 1:
            feature_path = matched_files[0].expanduser().absolute()
            if not (feature_path.suffix.lower() == ".npy" and feature_path.is_file()):
                raise ValueError(
                    f"Data file for the feature '{feature_name}' must be .npy. "
                    f"Path: {feature_path}"
                )
            elif not feature_path.is_file():
                raise RuntimeError(
                    f"Data file for the feature '{feature_name}' if not a file! "
                    f"Path: {feature_path}"
                )
            result[feature_name] = feature_path

    if len(result) == 0:
        # Return `None` if nothing is found.
        return None
    elif len(result) != len(column_to_filename_dict):
        # Raise an error if only some features are found!
        found_features = set(result.keys())
        all_features = set(column_to_filename_dict.keys())
        not_found_features = all_features.difference(found_features)
        raise ValueError(
            f"Data files for {len(not_found_features)} of "
            f"{len(all_features)} features not found! "
            f"Features: {not_found_features}"
        )
    return result


def is_local_numpy_dataset(
    dataset_folder: Union[str, Path], column_to_filename_dict: dict[str, str]
) -> bool:
    """Detects whether it's a numpy dataset saved as a collection of `.npy` files.

    In such datasets, each column is saved as a separate .npy file under
    the common root directory.

    Args:
        dataset_folder: Dataset location.
        column_to_filename_dict: A dictionary mapping column names
            to `.npy` file names (can be glob patterns).

    Returns:
        Whether a valid numpy dataset exists in this folder, and
        has data for all requested columns.
    """
    try:
        all_data_files: Optional[dict[str, Path]] = _try_resolve_column_numpy_files(
            dataset_folder, column_to_filename_dict, strict=False
        )
    except Exception:
        logger.exception("Invalid numpy dataset detected!")
        return False

    return all_data_files is not None


def load_local_numpy_dataset(
    dataset_folder: Union[str, Path], column_to_filename_dict: dict[str, str]
) -> pd.DataFrame:
    """Loads a numpy dataset saved as a collection of `.npy` files.

    Args:
        dataset_folder: Dataset location.
        column_to_filename_dict: A dictionary mapping column names
            to `.npy` file names (can be glob patterns).

    Returns:
        Pandas DataFrame.
    """
    all_data_files: Optional[dict[str, Path]] = _try_resolve_column_numpy_files(
        dataset_folder, column_to_filename_dict, strict=True
    )
    assert all_data_files is not None, "Unreachable if strict=True"

    data_columns: dict[str, np.ndarray] = {}
    for feature_name, data_file_path in all_data_files.items():
        column_data = np.load(data_file_path, allow_pickle=False)
        logger.info(
            f"Loaded '{feature_name}' (shape: {column_data.shape}) "
            f"from {data_file_path}"
        )
        data_columns[feature_name] = column_data

    return pd.DataFrame(data_columns)
