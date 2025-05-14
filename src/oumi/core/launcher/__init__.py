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

"""Launcher module for the Oumi (Open Universal Machine Intelligence) library.

This module provides base classes for cloud and cluster management in
the Oumi framework.

These classes serve as foundations for implementing cloud-specific and cluster-specific
launchers for running machine learning jobs.
"""

from oumi.core.launcher.base_cloud import BaseCloud
from oumi.core.launcher.base_cluster import BaseCluster, JobStatus

__all__ = [
    "BaseCloud",
    "BaseCluster",
    "JobStatus",
]
