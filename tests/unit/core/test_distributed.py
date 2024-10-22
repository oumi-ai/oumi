from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from oumi.core.distributed import (
    DeviceRankInfo,
    all_gather_object,
    estimate_dataloader_num_workers,
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
def mock_oumi_barrier():
    with patch("oumi.core.distributed.barrier") as mock_dist:
        yield mock_dist


@pytest.fixture
def mock_torch_barrier():
    with patch("torch.distributed.barrier") as mock:
        yield mock


@pytest.fixture
def mock_device_rank_info():
    with patch("oumi.core.distributed.get_device_rank_info") as mock_info:
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
    mock_oumi_barrier,
    mock_torch_barrier,
):
    # Disable distributed
    mock_torch_distributed.is_distributed.return_value = False

    # In non-distributed mode, the decorated function should be executed
    # exaclty once without calling any barrier
    with assert_function_called(mock_oumi_barrier, times=1):
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
    mock_oumi_barrier,
    mock_torch_barrier,
):
    # Disable distributed
    mock_torch_distributed.is_distributed.return_value = False

    # In non-distributed mode, the decorated function should be executed
    # exaclty once without calling any barrier
    with assert_function_called(mock_oumi_barrier, times=1):
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
    tested_decorator, mock_work_function, mock_oumi_barrier, mock_device_rank_info
):
    # Decorated function should be called by local leader
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=2, local_rank=0
    )
    assert is_local_process_zero() is True
    assert is_world_process_zero() is True

    with assert_function_called(mock_work_function, times=1):
        with assert_function_called(mock_oumi_barrier):

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
        with assert_function_called(mock_oumi_barrier):

            @tested_decorator()
            def test_function_should_not_execute():
                mock_work_function.do_work()
                pytest.fail("This should not be executed")

            test_function_should_not_execute()


def test_global_leader_only_should_do_work(
    mock_work_function, mock_oumi_barrier, mock_device_rank_info
):
    # Decorated function should be called by local leader
    # Barrier should be called ONCE
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=2, local_rank=0
    )
    assert is_local_process_zero() is True
    assert is_world_process_zero() is True

    with assert_function_called(mock_work_function, times=1):
        with assert_function_called(mock_oumi_barrier):

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
        with assert_function_called(mock_oumi_barrier):

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
        with assert_function_called(mock_oumi_barrier):

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
    mock_oumi_barrier,
    mock_torch_barrier,
):
    # The decorated function should be executed
    # exaclty once and call barrier exactly once
    # for both leaders and workers
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=2, local_rank=0
    )

    with assert_function_called(mock_oumi_barrier, times=1):
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

    with assert_function_called(mock_oumi_barrier, times=1):
        with assert_function_called(mock_work_function, times=1):

            @tested_decorator()
            def test_function():
                # This should be executed
                mock_work_function()

            test_function()


def test_estimate_dataloader_num_workers(mock_device_rank_info):
    assert estimate_dataloader_num_workers(gpus_per_node=0, cpu_count=32) == 1
    assert estimate_dataloader_num_workers(gpus_per_node=1, cpu_count=32) == 2
    assert estimate_dataloader_num_workers(gpus_per_node=2, cpu_count=32) == 4
    assert estimate_dataloader_num_workers(gpus_per_node=3, cpu_count=32) == 6
    assert estimate_dataloader_num_workers(gpus_per_node=4, cpu_count=32) == 8
    assert estimate_dataloader_num_workers(gpus_per_node=8, cpu_count=32) == 8

    assert estimate_dataloader_num_workers(gpus_per_node=0, cpu_count=8) == 1
    assert estimate_dataloader_num_workers(gpus_per_node=1, cpu_count=8) == 2
    assert estimate_dataloader_num_workers(gpus_per_node=2, cpu_count=8) == 2
    assert estimate_dataloader_num_workers(gpus_per_node=3, cpu_count=8) == 2
    assert estimate_dataloader_num_workers(gpus_per_node=4, cpu_count=8) == 2
    assert estimate_dataloader_num_workers(gpus_per_node=8, cpu_count=8) == 2

    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=8, rank=1, local_world_size=4, local_rank=0
    )
    assert estimate_dataloader_num_workers(None, cpu_count=32) == 8
    assert estimate_dataloader_num_workers(None, cpu_count=8) == 2

    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=2, rank=0, local_world_size=4, local_rank=3
    )
    assert estimate_dataloader_num_workers(None, cpu_count=32) == 8
    assert estimate_dataloader_num_workers(None, cpu_count=8) == 2


def test_all_gather_object_single_gpu(mock_device_rank_info, mock_torch_distributed):
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=1, rank=0, local_world_size=1, local_rank=0
    )

    with assert_function_called(mock_device_rank_info, times=2):
        assert all_gather_object({"aa": 12, "bb": 20}) == [{"aa": 12, "bb": 20}]


def test_all_gather_object_multi_gpu(mock_device_rank_info, mock_torch_distributed):
    mock_device_rank_info.return_value = DeviceRankInfo(
        world_size=4, rank=2, local_world_size=2, local_rank=0
    )

    def _all_gather_object_replicate(object_list, obj, group):
        for i in range(len(object_list)):
            object_list[i] = obj

    mock_torch_distributed.all_gather_object = MagicMock(
        side_effect=_all_gather_object_replicate
    )

    with assert_function_called(mock_device_rank_info, times=3):
        assert all_gather_object({"aa": 32, "bb": 40}) == [{"aa": 32, "bb": 40}] * 4

    assert mock_torch_distributed.is_available.call_count == 1
    assert mock_torch_distributed.is_initialized.call_count == 1
    assert mock_torch_distributed.all_gather_object.call_count == 1
