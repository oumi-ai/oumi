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

"""Test engine for validating analysis results."""

from oumi.analyze.testing.batch_engine import BatchTestEngine
from oumi.analyze.testing.engine import TestConfig, TestEngine
from oumi.analyze.testing.engine import TestType as TypedTestType
from oumi.analyze.testing.results import TestResult, TestSeverity, TestSummary

# Backward compatibility: API worker and batch engine import TestParams
# and TestType from here.  Re-export the core versions to keep existing
# ``from oumi.analyze.testing import TestType`` working.
from oumi.core.configs.params.test_params import TestParams, TestType

__all__ = [
    "BatchTestEngine",
    "TestConfig",
    "TestEngine",
    "TestParams",
    "TestResult",
    "TestSeverity",
    "TestSummary",
    "TestType",
    "TypedTestType",
]
