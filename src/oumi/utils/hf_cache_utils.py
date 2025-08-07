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

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HFCacheInfo
from huggingface_hub.utils import scan_cache_dir


@dataclass
class CachedItem:
    """A class representing a cached HuggingFace repository item."""

    repo_id: str
    size_bytes: int
    size: str
    repo_path: Path
    last_modified: str
    last_accessed: str
    repo_type: str
    nb_files: int


def list_hf_cache() -> list[CachedItem]:
    """List all cached items in the HuggingFace cache directory."""
    cache_dir: HFCacheInfo = scan_cache_dir()
    cached_items: list[CachedItem] = []
    for repo in cache_dir.repos:
        repo_id: str = repo.repo_id
        size_bytes: float = repo.size_on_disk
        repo_path: Path = repo.repo_path
        last_modified: str = repo.last_modified_str
        last_accessed: str = repo.last_accessed_str
        repo_type: str = repo.repo_type
        nb_files: int = repo.nb_files
        size: str = format_size(size_bytes)
        cached_items.append(
            CachedItem(
                repo_id=repo_id,
                size_bytes=size_bytes,
                size=size,
                repo_path=repo_path,
                last_modified=last_modified,
                last_accessed=last_accessed,
                repo_type=repo_type,
                nb_files=nb_files,
            )
        )
    return cached_items


def format_size(size_bytes: float) -> str:
    """Format bytes to human readable string."""
    # Simple implementation
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"
