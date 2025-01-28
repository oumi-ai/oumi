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

from transformers.models.phi3.configuration_phi3 import Phi3Config

# Consider using Cambrian clone of `modeling_phi3.py`,
# which has some XLA special cases.
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM, Phi3Model
