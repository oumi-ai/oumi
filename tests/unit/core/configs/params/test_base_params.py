from dataclasses import dataclass
from typing import Any

import pytest

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class NestedParams(BaseParams):
    simple_param: Any
    list_param: list
    dict_param: dict


@dataclass
class PositiveValueParams(BaseParams):
    value: Any

    def __finalize_and_validate__(self) -> None:
        if self.value <= 0:
            raise ValueError(f"Value must be positive, got {self.value}")


@dataclass
class ValueParams(BaseParams):
    value: Any


#
# Tests
#
def test_simple_params():
    simple = PositiveValueParams(42)
    simple.finalize_and_validate()  # Should not raise any exception


def test_nested_params():
    simple1 = PositiveValueParams(1)
    simple2 = PositiveValueParams(2)
    nested = NestedParams(
        simple_param=simple1,
        list_param=[simple2, "not a param"],
        dict_param={"a": PositiveValueParams(3), "b": "not a param"},
    )
    nested.finalize_and_validate()  # Should not raise any exception


def test_validation_call_count():
    @dataclass
    class CounterParams(BaseParams):
        call_count = 0
        a: Any = None
        b: Any = None
        c: Any = None

        def __finalize_and_validate__(self):
            CounterParams.call_count += 1

    root = CounterParams(
        a=CounterParams(),
        b=[CounterParams(), "not a param"],
        c={
            "x": CounterParams(),
            "y": "not a param",
        },
    )

    assert CounterParams.call_count == 0  # before validation
    root.finalize_and_validate()
    assert CounterParams.call_count == 4  # root + 3 unique nested CounterParams


def test_cyclic_reference():
    a = ValueParams(1)
    b = ValueParams(2)
    a.value = b
    b.value = a

    a.finalize_and_validate()  # Should not cause infinite recursion


def test_nested_params_with_failing_child():
    # Should not fail post init
    failing_param = PositiveValueParams(-5)

    # Should fail after validation
    with pytest.raises(expected_exception=ValueError, match="Value must be positive"):
        failing_param.finalize_and_validate()

    # Should fail after validation of parent
    nested = NestedParams(
        simple_param=PositiveValueParams(42),
        list_param=[
            PositiveValueParams(1),
            failing_param,
        ],
        dict_param={"a": PositiveValueParams(3), "b": PositiveValueParams(4)},
    )

    with pytest.raises(expected_exception=ValueError, match="Value must be positive"):
        nested.finalize_and_validate()
