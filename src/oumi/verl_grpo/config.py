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

Run with `python -m oumi.verl_grpo.config --config PATH --output PATH [KEY=VALUE ...]`
to write the verl config of an oumi training config.
"""

import argparse
import functools
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from omegaconf import DictConfig, OmegaConf, ValueNode
from omegaconf.errors import ConfigKeyError, ConfigTypeError

VERL_VERSION: Final = "0.9.1"
VERL_DEFAULTS_SHA256: Final = (
    "f19ebe24c2995052ca5a00de9c4ac302f6ba8c379effcc3ff83f058b0f9dcc90"
)
# Unedited copy of volcengine/verl v0.9.1
# `verl/trainer/config/_generated_ppo_trainer.yaml` (Apache-2.0).
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


def _check_shapes_survive(defaults: DictConfig, merged: DictConfig, path: str) -> None:
    """Checks that no layer turned a map into a non-map, or a scalar into a map.

    OmegaConf lets a scalar replace a map on merge, and lets a map replace a scalar
    below a struct node, so both are checked here. Interpolations are allowed to
    alias a whole section, and null defaults may take a map.
    """
    for key in defaults.keys():
        default_child = defaults._get_node(key)
        merged_child = merged._get_node(key)
        child_path = f"{path}.{key}" if path else str(key)
        if isinstance(default_child, DictConfig):
            if isinstance(merged_child, DictConfig):
                _check_shapes_survive(default_child, merged_child, child_path)
            elif not merged_child._is_interpolation():  # type: ignore[union-attr]
                raise ValueError(f"verl setting '{child_path}' must be a mapping.")
        elif (
            isinstance(default_child, ValueNode)
            and not default_child._is_none()
            and not default_child._is_interpolation()
            and isinstance(merged_child, DictConfig)
        ):
            raise ValueError(f"verl setting '{child_path}' must not be a mapping.")


def _find_list_over_map(
    defaults: DictConfig, layer: Mapping[Any, Any], path: str
) -> str | None:
    """Returns the dotted key of the first list that a layer puts over a map."""
    for key, value in layer.items():
        default_child = defaults._get_node(key) if key in defaults else None
        if not isinstance(default_child, DictConfig):
            continue
        child_path = f"{path}.{key}" if path else str(key)
        if isinstance(value, list):
            return child_path
        if isinstance(value, Mapping):
            found = _find_list_over_map(default_child, value, child_path)
            if found:
                return found
    return None


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
        ValueError: If a layer gives a setting the wrong shape, such as a scalar or
            list where verl has a map.
    """
    # The layers are combined before the struct merge so that a map set on a null
    # default accepts keys from a later layer, whatever way the layers are split.
    plain_layers = [
        OmegaConf.to_container(layer, resolve=False)
        if isinstance(layer, DictConfig)
        else dict(layer)
        for layer in layers
    ]
    combined = OmegaConf.to_container(OmegaConf.merge({}, *plain_layers), resolve=False)
    assert isinstance(combined, dict)
    defaults = _load_defaults()
    try:
        merged = OmegaConf.merge(defaults, combined)
    except ConfigKeyError as exc:
        raise UnknownVerlKeyError(str(exc.full_key)) from exc
    except ConfigTypeError as exc:
        key = _find_list_over_map(defaults, combined, "") or str(exc.full_key)
        raise ValueError(f"verl setting '{key}' has the wrong type.") from exc
    assert isinstance(merged, DictConfig)
    _check_shapes_survive(defaults, merged, "")
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=f"Write the verl {VERL_VERSION} config of an oumi training config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="oumi training config; its training.verl_config_overrides are merged "
        "onto verl's defaults.",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Where to write the verl config."
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="verl settings that win over the config's, such as data.train_files=PATH.",
    )
    args = parser.parse_args(argv)

    config_layer = OmegaConf.select(
        OmegaConf.load(args.config), "training.verl_config_overrides"
    )
    try:
        merged = merge_verl_config(
            config_layer or {}, OmegaConf.from_dotlist(args.overrides)
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
