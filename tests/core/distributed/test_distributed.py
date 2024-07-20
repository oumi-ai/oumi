from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lema.core.distributed import (
    DeviceRankInfo,
    global_leader_first,
    global_leader_only,
    is_local_process_zero,
    is_world_process_zero,
    local_leader_first,
    local_leader_only,
)


#
# Fixtures
#
@pytest.fixture
def mock_torch_distributed():
    with patch("torch.distributed") as mock_dist:
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        yield mock_dist


@pytest.fixture
def mock_lema_barrier():
    with patch("lema.core.distributed.barrier") as mock_dist:
        yield mock_dist


@pytest.fixture
def mock_torch_barrier():
    with patch("torch.distributed.barrier") as mock:
        yield mock


@pytest.fixture
def mock_device_rank_info():
    with patch("lema.core.distributed.get_device_rank_info") as mock_info:
        yield mock_info


@pytest.fixture
def mock_work_function():
    mock_work = MagicMock()
    yield mock_work


#
# Utils
#
@contextmanager
def assert_function_called(function, times=1):
    function.reset_mock()
    yield
    assert function.call_count == times


#
# Tests
#
@pytest.mark.parametrize(
    "tested_decorator",
    [local_leader_only, global_leader_only, local_leader_first, global_leader_first],
)
def test_decorators_without_distributed(
    tested_decorator,
    mock_work_function,
    mock_torch_distributed,
    mock_lema_barrier,
    mock_torch_barrier,
):
    # Disable distributed
    mock_torch_distributed.is_distributed.return_value = False

    # In non-distributed mode, the decorated function should be executed
    # exaclty once without calling any barrier
    with assert_function_called(mock_lema_barrier, times=1):
        with assert_function_called(mock_torch_barrier, times=0):
            with assert_function_called(mock_work_function, times=1):

                @tested_decorator()
                def test_function():
                    # This should be executed
                    mock_work_function()

                test_function()


@pytest.mark.parametrize(
    "decorator",
    [local_leader_first, global_leader_first],
)
def test_context_managers_without_distributed(
    decorator,
    mock_work_function,
    mock_torch_distributed,
    mock_lema_barrier,
    mock_torch_barrier,
):
    # Disable distributed
    mock_torch_distributed.is_distributed.return_value = False

    # In non-distributed mode, the decorated function should be executed
    # exaclty once without calling any barrier
    with assert_function_called(mock_lema_barrier, times=1):
        with assert_function_called(mock_torch_barrier, times=0):
            with assert_function_called(mock_work_function, times=1):
                with decorator():
                    # This should be executed
                    mock_work_function()


@pytest.mark.parametrize(
    "tested_decorator",
    [local_leader_only, global_leader_only],
)
def test_leaders_only_should_do_work(
    tested_decorator, mock_work_function, mock_lema_barrier, mock_device_rank_info
):
    # Decorated function should be called by local leader
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=2, local_rank=0
    )
    assert is_local_process_zero() is True
    assert is_world_process_zero() is True

    with assert_function_called(mock_work_function, times=1):
        with assert_function_called(mock_lema_barrier):

            @tested_decorator()
            def test_function_should_execute():
                mock_work_function()

            test_function_should_execute()

    # Decorated function should NOT called by other workers
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=1, local_world_size=2, local_rank=1
    )
    assert is_local_process_zero() is False
    assert is_world_process_zero() is False

    with assert_function_called(mock_work_function, times=0):
        with assert_function_called(mock_lema_barrier):

            @tested_decorator()
            def test_function_should_not_execute():
                mock_work_function.do_work()
                pytest.fail("This should not be executed")

            test_function_should_not_execute()


def test_global_leader_only_should_do_work(
    mock_work_function, mock_lema_barrier, mock_device_rank_info
):
    # Decorated function should be called by local leader
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=2, local_rank=0
    )
    assert is_local_process_zero() is True
    assert is_world_process_zero() is True

    with assert_function_called(mock_work_function, times=1):
        with assert_function_called(mock_lema_barrier):

            @global_leader_only()
            def test_function_should_execute():
                mock_work_function()

            test_function_should_execute()

    # Decorated function should NOT called by other local leaders
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=4, rank=3, local_world_size=2, local_rank=0
    )
    assert is_local_process_zero() is True
    assert is_world_process_zero() is False

    with assert_function_called(mock_work_function, times=0):
        with assert_function_called(mock_lema_barrier):

            @global_leader_only()
            def test_function_should_not_execute():
                mock_work_function.do_work()
                pytest.fail("This should not be executed")

            test_function_should_not_execute()

    # Decorated function should NOT called by other workers
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=1, local_world_size=2, local_rank=1
    )
    assert is_local_process_zero() is False
    assert is_world_process_zero() is False

    with assert_function_called(mock_work_function, times=0):
        with assert_function_called(mock_lema_barrier):

            @global_leader_only()
            def test_function_should_not_execute():
                mock_work_function.do_work()
                pytest.fail("This should not be executed")

            test_function_should_not_execute()


@pytest.mark.parametrize(
    "tested_decorator",
    [local_leader_first, global_leader_first],
)
def test_decorators_with_distributed(
    tested_decorator,
    mock_work_function,
    mock_lema_barrier,
    mock_torch_barrier,
):
    # The decorated function should be executed
    # exaclty once and call barrier exactly once
    # for both leaders and workers
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=2, local_rank=0
    )

    with assert_function_called(mock_lema_barrier, times=1):
        with assert_function_called(mock_work_function, times=1):

            @tested_decorator()
            def test_function():
                # This should be executed
                mock_work_function()

            test_function()

    # Worker node
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=1, local_world_size=2, local_rank=1
    )

    with assert_function_called(mock_lema_barrier, times=1):
        with assert_function_called(mock_work_function, times=1):

            @tested_decorator()
            def test_function():
                # This should be executed
                mock_work_function()

            test_function()
