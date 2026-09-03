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

"""Evaluation registry module."""

from oumi.evaluation.registry.berry_bench_task import berry_bench
from oumi.evaluation.registry.count_letters_task import count_letters
from oumi.evaluation.registry.healthbench_global_task import healthbench_global
from oumi.evaluation.registry.healthbench_task import healthbench
from oumi.evaluation.registry.rar_medicine_task import rar_medicine

__all__ = [
    "berry_bench",
    "count_letters",
    "healthbench",
    "healthbench_global",
    "rar_medicine",
]
