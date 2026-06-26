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

"""Stateful EHR tool executors.

``SyntheticEnvironment`` passes an isolated ``state`` snapshot (deep-copied
before each call). Executors must not mutate ``state`` in place.

Read-only executors return ``updated_state=None``. Write executors return a
new top-level state dict with the ``patients`` list rebuilt for the mutation;
the environment deep-copies ``updated_state`` before committing.

Recoverable errors (unknown patient, duplicate diagnosis, etc.) are returned
as structured ``{"status": "error", ...}`` payloads in ``output``, not raised.
"""

from oumi.examples.ehr.executors.add_diagnosis import add_diagnosis
from oumi.examples.ehr.executors.get_patient import get_patient
from oumi.examples.ehr.executors.list_patients import list_patients
from oumi.examples.ehr.executors.prescribe_medication import prescribe_medication
from oumi.examples.ehr.executors.record_vitals import record_vitals
from oumi.examples.ehr.executors.update_allergies import update_allergies

__all__ = [
    "add_diagnosis",
    "get_patient",
    "list_patients",
    "prescribe_medication",
    "record_vitals",
    "update_allergies",
]
