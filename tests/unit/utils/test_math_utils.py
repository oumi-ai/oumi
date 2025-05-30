from oumi.utils.math_utils import (
    is_power_of_two,
)


def test_is_power_of_two_basic():
    """Test the is_power_of_two function."""
    assert is_power_of_two(1)
    assert is_power_of_two(2)
    assert is_power_of_two(4)
    assert is_power_of_two(8)
    assert is_power_of_two(16)
    assert is_power_of_two(2147483648)

    assert not is_power_of_two(3)
    assert not is_power_of_two(5)
    assert not is_power_of_two(6)
    assert not is_power_of_two(7)
    assert not is_power_of_two(0)

    # Negative powers of two are considered valid
    assert is_power_of_two(-1)
    assert is_power_of_two(-2)
    assert is_power_of_two(-256)
    assert is_power_of_two(-2147483648)
