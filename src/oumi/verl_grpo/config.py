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

"""Strict merging of verl settings onto the pinned verl defaults.

This module imports only omegaconf and the standard library, so it can be used
from environments that have neither verl nor Oumi's trainer stack installed.
"""

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError

VERL_VERSION: Final = "0.9.1"
VERL_DEFAULTS_SHA256: Final = (
    "f19ebe24c2995052ca5a00de9c4ac302f6ba8c379effcc3ff83f058b0f9dcc90"
)
VERL_DEFAULTS_FILE: Final[Path] = (
    Path(__file__).parent / f"ppo_trainer_defaults_{VERL_VERSION}.yaml"
)


class UnknownVerlKeyError(ValueError):
    """A setting that verl's defaults don't define."""

    def __init__(self, key: str):
        """Initializes the error with the dotted path of the unknown key."""
        super().__init__(f"verl {VERL_VERSION} has no setting '{key}'")
        self.key = key


def _open_free_form_maps(node: DictConfig) -> None:
    """Lets maps that are empty in the defaults accept arbitrary new keys."""
    if len(node) == 0:
        OmegaConf.set_struct(node, False)
        return
    for key in node.keys():
        child = node._get_node(key)
        if isinstance(child, DictConfig):
            _open_free_form_maps(child)


def _check_maps_stay_maps(defaults: DictConfig, merged: DictConfig, path: str) -> None:
    """OmegaConf lets a scalar replace a map on merge; verl's settings can't."""
    for key in defaults.keys():
        default_child = defaults._get_node(key)
        if not isinstance(default_child, DictConfig):
            continue
        child_path = f"{path}.{key}" if path else str(key)
        merged_child = merged._get_node(key)
        if not isinstance(merged_child, DictConfig):
            raise ValueError(f"verl setting '{child_path}' must be a mapping.")
        _check_maps_stay_maps(default_child, merged_child, child_path)


@functools.cache
def _load_defaults() -> DictConfig:
    defaults = OmegaConf.load(VERL_DEFAULTS_FILE)
    assert isinstance(defaults, DictConfig)
    # verl reads this map but leaves it out of its defaults.
    OmegaConf.update(
        defaults, "reward.custom_reward_function.reward_kwargs", {}, force_add=True
    )
    OmegaConf.set_struct(defaults, True)
    _open_free_form_maps(defaults)
    return defaults


def merge_verl_config(*layers: Mapping[str, Any] | DictConfig) -> DictConfig:
    """Merges layers onto the pinned verl defaults; later layers win.

    Only maps that are empty in verl's defaults, and `reward_kwargs`, accept keys
    verl doesn't define. Interpolations are left unresolved.

    Args:
        layers: Settings to merge, in increasing precedence.

    Returns:
        A new, writable, struct-mode config.

    Raises:
        UnknownVerlKeyError: If a layer sets a key verl doesn't define.
        ValueError: If a layer puts a scalar where verl has a map.
    """
    plain_layers = [
        OmegaConf.to_container(layer, resolve=False)
        if isinstance(layer, DictConfig)
        else layer
        for layer in layers
    ]
    defaults = _load_defaults()
    try:
        merged = OmegaConf.merge(defaults, *plain_layers)
    except ConfigKeyError as exc:
        raise UnknownVerlKeyError(str(exc.full_key)) from exc
    assert isinstance(merged, DictConfig)
    _check_maps_stay_maps(defaults, merged, "")
    return merged
