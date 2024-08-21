from unittest.mock import patch

import pytest

from lema.core.configs import ModelParams
from lema.core.types import HardwareException


def test_flash_attention_hardware_check():
    # flash_attention_2 is requested and available
    with patch(
        "lema.core.configs.params.model_params.is_flash_attn_2_available",
        return_value=True,
    ):
        config = ModelParams()
        config.attn_implementation = "flash_attention_2"
        config.validate()  # Should NOT raise an exception

    # flash_attention_2 is NOT requested
    config = ModelParams()
    config.attn_implementation = "other_implementation"
    config.validate()  # Should NOT raise an exception

    # flash_attention_2 is requested but is NOT available
    with patch(
        "lema.core.configs.params.model_params.is_flash_attn_2_available",
        return_value=False,
    ):
        config = ModelParams()
        config.attn_implementation = "flash_attention_2"
        with pytest.raises(
            HardwareException,
            match="Flash attention 2 was requested but it is not supported",
        ):
            config.validate()
