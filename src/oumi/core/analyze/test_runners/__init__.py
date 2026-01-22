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

"""Test runners for dataset analysis quality tests.

This package provides test runner implementations for different test types.
Each runner knows how to execute a specific type of test on analysis DataFrames.
"""

from oumi.core.analyze.test_runners.base import BaseTestRunner
from oumi.core.analyze.test_runners.composite import CompositeTestRunner
from oumi.core.analyze.test_runners.contains import ContainsTestRunner
from oumi.core.analyze.test_runners.distribution import DistributionTestRunner
from oumi.core.analyze.test_runners.outliers import OutliersTestRunner
from oumi.core.analyze.test_runners.percentage import PercentageTestRunner
from oumi.core.analyze.test_runners.python_runner import PythonTestRunner
from oumi.core.analyze.test_runners.query import QueryTestRunner
from oumi.core.analyze.test_runners.regex import RegexTestRunner
from oumi.core.analyze.test_runners.threshold import ThresholdTestRunner

__all__ = [
    "BaseTestRunner",
    "CompositeTestRunner",
    "ContainsTestRunner",
    "DistributionTestRunner",
    "OutliersTestRunner",
    "PercentageTestRunner",
    "PythonTestRunner",
    "QueryTestRunner",
    "RegexTestRunner",
    "ThresholdTestRunner",
]
