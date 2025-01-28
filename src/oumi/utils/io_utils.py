# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
from typing import Any, Union

import jsonlines


def load_json(filename: Union[str, Path]) -> Any:
    """Load JSON data from a file.

    Args:
        filename: Path to the JSON file.

    Returns:
        dict: Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    data: dict[str, Any], filename: Union[str, Path], indent: int = 2
) -> None:
    """Save data as a formatted JSON file.

    Args:
        data: The data to be saved as JSON.
        filename: Path where the JSON file will be saved.
        indent: Number of spaces for indentation. Defaults to 2.

    Raises:
        TypeError: If the data is not JSON serializable.
    """
    file_path = Path(filename)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=indent, ensure_ascii=False)


def load_file(filename: Union[str, Path], encoding: str = "utf-8") -> str:
    """Load a file as a string.

    Args:
        filename: Path to the file.
        encoding: Encoding to use when reading the file. Defaults to "utf-8".

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    file_path = Path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    with file_path.open("r", encoding=encoding) as file:
        return file.read()


def get_oumi_root_directory() -> Path:
    """Get the root directory of the Oumi project.

    Returns:
        Path: The absolute path to the Oumi project's root directory.
    """
    return Path(__file__).parent.parent.resolve()


def load_jsonlines(filename: Union[str, Path]) -> list[dict[str, Any]]:
    """Load a jsonlines file.

    Args:
        filename: Path to the jsonlines file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a
            JSON object from the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        jsonlines.InvalidLineError: If the file contains invalid JSON.
    """
    file_path = Path(filename)

    if file_path.is_dir():
        raise ValueError(
            f"Provided path is a directory, expected a file: '{filename}'."
        )

    if not file_path.is_file():
        raise FileNotFoundError(f"Provided path does not exist: '{filename}'.")

    with jsonlines.open(file_path) as reader:
        return list(reader)


def save_jsonlines(filename: Union[str, Path], data: list[dict[str, Any]]) -> None:
    """Save a list of dictionaries to a jsonlines file.

    Args:
        filename: Path to the jsonlines file to be created or overwritten.
        data: A list of dictionaries to be saved as JSON objects.

    Raises:
        IOError: If there's an error writing to the file.
    """
    file_path = Path(filename)

    try:
        with jsonlines.open(file_path, mode="w") as writer:
            writer.write_all(data)
    except OSError as e:
        raise OSError(f"Error writing to file {filename}") from e
