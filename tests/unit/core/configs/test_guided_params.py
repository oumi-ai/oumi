import pytest

from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams


def test_guided_decoding_params_mutually_exclusive():
    """Test that json, regex, and choice parameters are mutually exclusive."""
    # Valid cases - only one or none specified
    GuidedDecodingParams(json={"type": "object"})
    GuidedDecodingParams(regex=r"\d+")
    GuidedDecodingParams(choice=["option1", "option2"])
    GuidedDecodingParams()  # None specified

    # Invalid cases - multiple specified
    error_msg = "Only one of 'json', 'regex', or 'choice' can be specified"

    with pytest.raises(ValueError, match=error_msg):
        GuidedDecodingParams(json={"type": "object"}, regex=r"\d+")

    with pytest.raises(ValueError, match=error_msg):
        GuidedDecodingParams(json={"type": "object"}, choice=["option1"])

    with pytest.raises(ValueError, match=error_msg):
        GuidedDecodingParams(regex=r"\d+", choice=["option1"])

    with pytest.raises(ValueError, match=error_msg):
        GuidedDecodingParams(json={"type": "object"}, regex=r"\d+", choice=["option1"])
