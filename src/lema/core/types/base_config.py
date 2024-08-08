import dataclasses
import logging
import re
from typing import Any, Iterator, List, Optional, Tuple, Type, TypeVar, cast

from omegaconf import OmegaConf

from lema.core.types.params.base_params import BaseParams

T = TypeVar("T", bound="BaseConfig")

_CLI_IGNORED_PREFIXES = ["--local-rank"]


def _filter_ignored_args(arg_list: List[str]) -> List[str]:
    """Filters out ignored CLI arguments."""
    return [
        arg
        for arg in arg_list
        if not any(arg.startswith(prefix) for prefix in _CLI_IGNORED_PREFIXES)
    ]


@dataclasses.dataclass
class BaseConfig:
    def to_yaml(self, config_path: str) -> None:
        """Saves the configuration to a YAML file."""
        OmegaConf.save(config=self, f=config_path)

    @classmethod
    def from_yaml(cls: Type[T], config_path: str, ignore_interpolation=True) -> T:
        """Loads a configuration from a YAML file.

        Args:
            config_path: The path to the YAML file.
            ignore_interpolation: If True, then any interpolation variables in the
                configuration file will be escaped.

        Returns:
            BaseConfig: The merged configuration object.
        """
        schema = OmegaConf.structured(cls)
        if ignore_interpolation:
            with open(config_path) as f:
                stringified_config = f.read()
                pattern = r"(?<!\\)\$\{"  # Matches "${" but not "\${"
                stringified_config = re.sub(pattern, "\\${", stringified_config)
                file_config = OmegaConf.create(stringified_config)
        else:
            file_config = OmegaConf.load(config_path)
        config = OmegaConf.to_object(OmegaConf.merge(schema, file_config))
        if not isinstance(config, cls):
            raise TypeError(f"config is not {cls}")
        return cast(cls, config)

    @classmethod
    def from_yaml_and_arg_list(
        cls: Type[T],
        config_path: Optional[str],
        arg_list: List[str],
        logger: Optional[logging.Logger] = None,
    ) -> T:
        """Loads a configuration from various sources.

        If both YAML and arguments list are provided, then
        parameters specified in `arg_list` have higher precedence.

        Args:
            config_path: The path to the YAML file.
            arg_list: Command line arguments list.
            logger: (optional) Logger.

        Returns:
            BaseConfig: The merged configuration object.
        """
        # Start with an empty typed config. This forces OmegaConf to validate
        # that all other configs are of this structured type as well.
        all_configs = [OmegaConf.structured(cls)]

        # Override with configuration file if provided.
        if config_path is not None:
            all_configs.append(cls.from_yaml(config_path))

        # Filter out CLI arguments that should be ignored.
        arg_list = _filter_ignored_args(arg_list)

        # Override with CLI arguments.
        all_configs.append(OmegaConf.from_cli(arg_list))
        try:
            # Merge and validate configs
            config = OmegaConf.merge(*all_configs)
        except Exception:
            if logger:
                logger.exception(f"Failed to merge Omega configs: {all_configs}")
            raise

        config = OmegaConf.to_object(config)
        if not isinstance(config, cls):
            raise TypeError(f"config {type(config)} is not {type(cls)}")

        return cast(cls, config)

    def validate(self) -> None:
        """Validates the top level params objects."""
        for _, attr_value in self:
            if isinstance(attr_value, BaseParams):
                attr_value.validate()

        self.__validate__()

    def __validate__(self) -> None:
        """Validates the parameters of this object.

        This method can be overridden by subclasses to implement custom
        validation logic.

        In case of validation errors, this method should raise a `ValueError`
        or other appropriate exception.
        """

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Returns an iterator over field names and values.

        Note: for an attribute to be a field, it must be declared in the
        dataclass definition and have a type annotation.
        """
        for param in dataclasses.fields(self):
            yield param.name, getattr(self, param.name)
