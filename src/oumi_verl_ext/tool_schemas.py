# Copyright 2026 - Oumi
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

"""The tool config file's full tool schemas, for the prompt and verl's length filter.

verl imports it in its processes, so it imports only verl, the OmegaConf verl reads
its configs with, the standard library and ``oumi_verl_ext``.
"""

import functools
from types import ModuleType
from typing import Any

from omegaconf import OmegaConf
from verl.utils.import_utils import (  # pyright: ignore[reportMissingImports]
    resolve_config_path,
)

from oumi_verl_ext.tool_episode import full_tool_schemas


def with_full_schemas(
    schemas: list[dict[str, Any]] | None, tool_config_path: str | None
) -> list[dict[str, Any]] | None:
    """``schemas`` with the full tool schemas from ``tool_config_path``, if set."""
    if schemas is None or not tool_config_path:
        return schemas
    return full_tool_schemas(_config_schemas(tool_config_path), schemas)


@functools.cache
def _config_schemas(tool_config_path: str) -> list[Any]:
    # Reads the file as verl's initialize_tools_from_config does, once per process,
    # since verl builds a loop per trajectory.
    tools = OmegaConf.load(resolve_config_path(tool_config_path)).tools
    return [
        OmegaConf.to_container(tool.tool_schema, resolve=True)
        for tool in tools
        if tool.get("tool_schema") is not None
    ]


def patch_length_filter(rl_dataset: ModuleType) -> None:
    """Makes verl's prompt-length filter count the full tool schemas."""
    filter_long_prompts = rl_dataset.RLHFDataset.maybe_filter_out_long_prompts

    def maybe_filter_out_long_prompts(self, dataframe=None):
        self.tool_schemas = with_full_schemas(self.tool_schemas, self.tool_config_path)
        return filter_long_prompts(self, dataframe)

    rl_dataset.RLHFDataset.maybe_filter_out_long_prompts = maybe_filter_out_long_prompts
