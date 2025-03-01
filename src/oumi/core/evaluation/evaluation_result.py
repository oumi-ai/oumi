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

from typing import Any, Optional


class EvaluationResult:
    def __init__(
        self,
        task_name: Optional[str] = None,
        task_result: Optional[dict[str, Any]] = None,
        backend_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the EvaluationResult class."""
        self.task_name = task_name
        self.task_result = task_result or {}
        self.backend_config = backend_config or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert the EvaluationResult to a dictionary."""
        return {
            "task_name": self.task_name,
            "task_result": self.task_result,
            "backend_config": self.backend_config,
        }
