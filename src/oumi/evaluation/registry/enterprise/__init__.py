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

"""Enterprise evaluation functions for SFT task evaluation.

This module provides custom evaluation functions for enterprise tasks:
- Banking77: 77-class classification accuracy
- PubMedQA: 3-class classification accuracy
- TAT-QA: Exact match and F1 for tabular QA
- NL2SQL: Edit distance for SQL generation
- SimpleSafetyTests: Safety evaluation with refusal detection
"""

# Import to register evaluation functions
from oumi.evaluation.registry.enterprise.classification import (
    enterprise_banking77,
    enterprise_pubmedqa,
)
from oumi.evaluation.registry.enterprise.nl2sql import enterprise_nl2sql
from oumi.evaluation.registry.enterprise.simple_safety_tests import simple_safety_tests
from oumi.evaluation.registry.enterprise.tatqa import enterprise_tatqa

__all__ = [
    "enterprise_banking77",
    "enterprise_pubmedqa",
    "enterprise_tatqa",
    "enterprise_nl2sql",
    "simple_safety_tests",
]
