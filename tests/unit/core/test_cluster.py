# Copyright 2025 - Oumi
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

"""Tests for oumi.core.cluster module."""

import pytest

from oumi.core.cluster import (
    ClusterInfo,
    WorldInfo,
)


class TestWorldInfo:
    """Tests for WorldInfo namedtuple."""

    def test_world_info_creation(self):
        """Test creating a WorldInfo."""
        info = WorldInfo(num_nodes=2, gpus_per_node=4)
        assert info.num_nodes == 2
        assert info.gpus_per_node == 4


class TestClusterInfo:
    """Tests for ClusterInfo class."""

    def test_cluster_info_creation(self):
        """Test creating a valid ClusterInfo."""
        info = ClusterInfo(
            node_rank=0,
            world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
            master_address="192.168.1.1",
            master_port=8007,
            node_ips=["192.168.1.1", "192.168.1.2"],
        )
        assert info.node_rank == 0
        assert info.num_nodes == 2
        assert info.gpus_per_node == 4
        assert info.total_gpus == 8
        assert info.master_address == "192.168.1.1"
        assert info.master_port == 8007
        assert info.node_ips == ["192.168.1.1", "192.168.1.2"]

    def test_cluster_info_invalid_nodes(self):
        """Test that invalid num_nodes raises ValueError."""
        with pytest.raises(ValueError, match="Non-positive number of nodes"):
            ClusterInfo(
                node_rank=0,
                world_info=WorldInfo(num_nodes=0, gpus_per_node=4),
                master_address="192.168.1.1",
                master_port=8007,
                node_ips=["192.168.1.1"],
            )

    def test_cluster_info_invalid_gpus(self):
        """Test that invalid gpus_per_node raises ValueError."""
        with pytest.raises(ValueError, match="Non-positive number of nodes"):
            ClusterInfo(
                node_rank=0,
                world_info=WorldInfo(num_nodes=2, gpus_per_node=0),
                master_address="192.168.1.1",
                master_port=8007,
                node_ips=["192.168.1.1", "192.168.1.2"],
            )

    def test_cluster_info_invalid_node_rank_negative(self):
        """Test that negative node_rank raises ValueError."""
        with pytest.raises(ValueError, match="Node rank -1 is out of range"):
            ClusterInfo(
                node_rank=-1,
                world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
                master_address="192.168.1.1",
                master_port=8007,
                node_ips=["192.168.1.1", "192.168.1.2"],
            )

    def test_cluster_info_invalid_node_rank_too_high(self):
        """Test that node_rank >= num_nodes raises ValueError."""
        with pytest.raises(ValueError, match="Node rank 2 is out of range"):
            ClusterInfo(
                node_rank=2,
                world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
                master_address="192.168.1.1",
                master_port=8007,
                node_ips=["192.168.1.1", "192.168.1.2"],
            )

    def test_cluster_info_empty_master_address(self):
        """Test that empty master_address raises ValueError."""
        with pytest.raises(ValueError, match="Empty master address"):
            ClusterInfo(
                node_rank=0,
                world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
                master_address="",
                master_port=8007,
                node_ips=["192.168.1.1", "192.168.1.2"],
            )

    def test_cluster_info_invalid_port_too_low(self):
        """Test that port < 1024 raises ValueError."""
        with pytest.raises(ValueError, match="Master port: 100 is outside of valid range"):
            ClusterInfo(
                node_rank=0,
                world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
                master_address="192.168.1.1",
                master_port=100,
                node_ips=["192.168.1.1", "192.168.1.2"],
            )

    def test_cluster_info_invalid_port_too_high(self):
        """Test that port > 65535 raises ValueError."""
        with pytest.raises(ValueError, match="Master port: 70000 is outside of valid range"):
            ClusterInfo(
                node_rank=0,
                world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
                master_address="192.168.1.1",
                master_port=70000,
                node_ips=["192.168.1.1", "192.168.1.2"],
            )

    def test_cluster_info_repr(self):
        """Test ClusterInfo string representation."""
        info = ClusterInfo(
            node_rank=1,
            world_info=WorldInfo(num_nodes=2, gpus_per_node=4),
            master_address="192.168.1.1",
            master_port=8007,
            node_ips=["192.168.1.1", "192.168.1.2"],
        )
        repr_str = repr(info)
        assert "node_rank" in repr_str
        assert "1" in repr_str
        assert "num_nodes" in repr_str
        assert "2" in repr_str
