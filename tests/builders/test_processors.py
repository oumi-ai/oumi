import pytest

from oumi.builders.processors import build_processor


@pytest.mark.parametrize(
    "trust_remote_code",
    [
        False,
        True,
    ],
)
def test_build_processor_empty_name(trust_remote_code):
    with pytest.raises(ValueError, match="Empty model name"):
        build_processor("", trust_remote_code=trust_remote_code)
