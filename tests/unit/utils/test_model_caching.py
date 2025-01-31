import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from oumi.utils.model_caching import get_local_filepath_for_gguf

HF_REPO_ID = "repo_id"
HF_GGUF_FILENAME = "file.gguf"


def _mock_download(repo_id: str, filename: str, local_dir: str) -> str:
    gguf_file_path = Path(local_dir) / filename
    if not gguf_file_path.exists():
        with open(gguf_file_path, "w") as file:
            file.write("downloaded GGUF")
    return gguf_file_path.absolute().as_posix()


def test_caching_gguf():
    with (
        tempfile.TemporaryDirectory() as output_top_dir,
        patch(
            "oumi.utils.model_caching.hf_hub_download", side_effect=_mock_download
        ) as mock_download,
    ):
        output_folder = os.path.join(output_top_dir, "subfolder")
        output_file = os.path.join(output_folder, HF_GGUF_FILENAME)
        assert not os.path.exists(output_folder)

        # Download the file and check that it is in the expected location.
        gguf_path = get_local_filepath_for_gguf(
            repo_id=HF_REPO_ID,
            filename=HF_GGUF_FILENAME,
            cache_dir=output_folder,
        )
        assert gguf_path == output_file
        assert os.path.exists(output_file)
        with open(output_file) as file:
            assert file.read() == "downloaded GGUF"

        # Get the file from cache.
        gguf_path = get_local_filepath_for_gguf(
            repo_id=HF_REPO_ID,
            filename=HF_GGUF_FILENAME,
            cache_dir=output_folder,
        )
        assert gguf_path == output_file
        assert os.path.exists(output_file)
        with open(output_file) as file:
            assert file.read() == "downloaded GGUF"

        # Make sure the download happened once (the 2nd time we cached the GGUF).
        mock_download.assert_called_once_with(
            repo_id=HF_REPO_ID,
            filename=HF_GGUF_FILENAME,
            local_dir=output_folder,
        )


def test_caching_gguf_invalid_filename():
    with tempfile.TemporaryDirectory() as output_folder:
        with pytest.raises(
            ValueError,
            match="The `filename` provided is not a `.gguf` file: `invalid_file.txt`",
        ):
            get_local_filepath_for_gguf(
                repo_id=HF_REPO_ID,
                filename="invalid_file.txt",  # Invalid: not a GGUF file.
                cache_dir=output_folder,
            )
