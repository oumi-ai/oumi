"""Launcher module for the Oumi (Open Unified Machine Intelligence) library.

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
