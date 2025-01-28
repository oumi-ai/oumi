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

from pathlib import Path

from huggingface_hub import hf_hub_download

from oumi.utils.logging import logger

HUGGINGFACE_CACHE = ".cache/huggingface"


def get_local_filepath_for_gguf(
    repo_id: str, filename: str, cache_dir=HUGGINGFACE_CACHE
) -> str:
    """Return a local path for the provided GGUF file, downloading it if necessary.

    Args:
        repo_id: HuggingFace Hub repo ID (e.g., `bartowski/Llama-3.2-3B-Instruct-GGUF`)
        filename: HuggingFace Hub filename (e.g., `Llama-3.2-3B-Instruct-Q8_0.gguf`)
        cache_dir: Local path to cached models. Defaults to `HUGGINGFACE_CACHE`.

    Returns:
        A local path caching the GGUF file.
    """
    # Ensure that the filename corresponds to a `GGUF` file.
    if Path(filename).suffix != ".gguf":
        raise ValueError(f"The `filename` provided is not a `.gguf` file: `{filename}`")

    # Ensure the cache directory exists. If not, create it.
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    gguf_local_file_path = cache_dir / filename

    # Check if the file is already cached; if not, download it.
    if gguf_local_file_path.exists():
        logger.info(f"Loading GGUF file from cache ({str(gguf_local_file_path)}).")
        return gguf_local_file_path.absolute().as_posix()
    else:
        logger.info(f"Downloading GGUF file `{filename}` from HuggingFace.")
        try:
            gguf_local_file_path = Path(
                hf_hub_download(
                    repo_id=repo_id, filename=filename, local_dir=cache_dir.as_posix()
                )
            )
        except Exception:
            logger.exception(
                f"Failed to download the GGUF file `{filename}` from HuggingFace "
                f"Hub's repo id `{repo_id}`."
            )
        return gguf_local_file_path.absolute().as_posix()
