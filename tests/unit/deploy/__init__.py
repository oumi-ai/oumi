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

"""MCP prompt templates for Oumi training workflows."""

from oumi.mcp.prompts.analyze import ANALYZE_COMMAND_RESOURCE
from oumi.mcp.prompts.cloud_launch import CLOUD_LAUNCH_RESOURCE
from oumi.mcp.prompts.eval import EVAL_COMMAND_RESOURCE
from oumi.mcp.prompts.get_started import GET_STARTED_CONTENT
from oumi.mcp.prompts.infer import INFER_COMMAND_RESOURCE
from oumi.mcp.prompts.mle_workflow import MLE_WORKFLOW_RESOURCE
from oumi.mcp.prompts.post_training import POST_TRAINING_RESOURCE
from oumi.mcp.prompts.synth import SYNTH_COMMAND_RESOURCE
from oumi.mcp.prompts.train import TRAIN_COMMAND_RESOURCE

__all__ = [
    "ANALYZE_COMMAND_RESOURCE",
    "CLOUD_LAUNCH_RESOURCE",
    "EVAL_COMMAND_RESOURCE",
    "GET_STARTED_CONTENT",
    "INFER_COMMAND_RESOURCE",
    "MLE_WORKFLOW_RESOURCE",
    "POST_TRAINING_RESOURCE",
    "SYNTH_COMMAND_RESOURCE",
    "TRAIN_COMMAND_RESOURCE",
]
