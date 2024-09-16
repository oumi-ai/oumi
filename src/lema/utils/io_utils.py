import json
from pathlib import Path
from typing import Any, Dict, Union


def load_json(filename: Union[str, Path]) -> Dict[str, Any]:
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
    data: Dict[str, Any], filename: Union[str, Path], indent: int = 2
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
