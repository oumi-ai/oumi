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

"""Model management command handler."""

import os
from pathlib import Path
from typing import Any

from rich.panel import Panel

from oumi.core.commands.base_handler import BaseCommandHandler, CommandResult
from oumi.core.commands.command_parser import ParsedCommand


class ModelManagementHandler(BaseCommandHandler):
    """Handles model-related commands: swap, list_engines."""

    def get_supported_commands(self) -> list[str]:
        """Get list of commands this handler supports."""
        return ["swap", "list_engines"]

    def handle_command(self, command: ParsedCommand) -> CommandResult:
        """Handle a model management command."""
        if command.command == "swap":
            return self._handle_swap(command)
        elif command.command == "list_engines":
            return self._handle_list_engines(command)
        else:
            return CommandResult(
                success=False,
                message=f"Unsupported command: {command.command}",
                should_continue=False,
            )

    def _handle_swap(self, command: ParsedCommand) -> CommandResult:
        """Handle the /swap(model_name) or /swap(config:path) command to switch models while preserving conversation."""
        try:
            if not command.args:
                return CommandResult(
                    success=False,
                    message="swap command requires a model name or config path argument",
                    should_continue=False,
                )

            target = command.args[0].strip()

            # Check if this is a config-based swap
            if target.startswith("config:"):
                config_path = target[7:]  # Remove "config:" prefix
                return self._handle_config_swap(config_path)

            # Regular model swap
            # Save current model state to branch if branch manager available
            if hasattr(self.context, "branch_manager") and self.context.branch_manager:
                current_branch = self.context.branch_manager.get_current_branch()
                if current_branch:
                    self._save_current_model_state_to_branch(current_branch["id"])

            # For now, return a placeholder message since actual model swapping
            # requires infrastructure changes
            return CommandResult(
                success=False,
                message=(
                    f"Model swapping to '{target}' is not yet implemented. "
                    "This feature requires infrastructure support for dynamic model loading."
                ),
                should_continue=False,
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error swapping model: {str(e)}",
                should_continue=False,
            )

    def _handle_config_swap(self, config_path: str) -> CommandResult:
        """Handle config-based model swapping by loading an Oumi YAML config."""
        try:
            # Resolve config path
            if not os.path.isabs(config_path):
                # Try relative to current directory first
                full_path = Path.cwd() / config_path
                if not full_path.exists():
                    # Try relative to config directory in project
                    project_config_path = (
                        Path(__file__).parent.parent.parent.parent.parent
                        / "configs"
                        / config_path
                    )
                    if project_config_path.exists():
                        full_path = project_config_path
                    else:
                        return CommandResult(
                            success=False,
                            message=f"Config file not found: {config_path}",
                            should_continue=False,
                        )
            else:
                full_path = Path(config_path)

            if not full_path.exists():
                return CommandResult(
                    success=False,
                    message=f"Config file not found: {config_path}",
                    should_continue=False,
                )

            # Load and parse the new config
            from oumi.core.configs import InferenceConfig
            
            try:
                new_config = InferenceConfig.from_yaml(str(full_path))
            except Exception as e:
                return CommandResult(
                    success=False,
                    message=f"Error loading config: {str(e)}",
                    should_continue=False,
                )

            # Create new inference engine with the loaded config
            from oumi.core.inference.factory import create_inference_engine
            
            try:
                new_engine = create_inference_engine(new_config)
                
                # Replace the current inference engine and config
                self.context.inference_engine = new_engine
                self.context.config = new_config
                
                # Update system monitor with new model info if available
                if hasattr(self.context, 'system_monitor') and self.context.system_monitor:
                    max_context = getattr(new_config.model, 'model_max_length', 4096)
                    if hasattr(self.context.system_monitor, 'update_max_context_tokens'):
                        self.context.system_monitor.update_max_context_tokens(max_context)
                
                model_name = getattr(new_config.model, 'model_name', 'Unknown model')
                engine_type = getattr(new_config, 'engine', 'Unknown engine')
                
                return CommandResult(
                    success=True,
                    message=f"✅ Swapped to {model_name} using {engine_type} engine",
                    should_continue=False,
                )
                
            except Exception as e:
                return CommandResult(
                    success=False,
                    message=f"Error creating inference engine: {str(e)}",
                    should_continue=False,
                )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error with config swap: {str(e)}",
                should_continue=False,
            )

    def _handle_list_engines(self, command: ParsedCommand) -> CommandResult:
        """Handle the /list_engines() command to list available inference engines and sample models."""
        try:
            # Get style attributes
            use_emoji = getattr(self._style, "use_emoji", True)
            title_style = getattr(self._style, "assistant_title_style", "bold cyan")

            # Get engines information
            engines_info = self._get_engines_info()

            # Build output
            output_lines = []
            if use_emoji:
                output_lines.append("🔧 **Available Inference Engines**\\n")
            else:
                output_lines.append("**Available Inference Engines**\\n")

            for engine_info in engines_info:
                engine_name = engine_info["name"]
                engine_type = engine_info["type"]
                description = engine_info["description"]
                sample_models = engine_info["sample_models"]
                api_key_required = engine_info.get("api_key_required", False)

                # Engine header
                type_emoji = {"Local": "💻", "API": "🌐", "Remote": "🔗"}.get(
                    engine_type, "⚙️"
                )
                output_lines.append(
                    f"\\n### {type_emoji} {engine_name} ({engine_type})"
                )
                output_lines.append(f"{description}")

                if api_key_required:
                    output_lines.append("🔑 **Requires API Key**")

                if sample_models:
                    output_lines.append("\\n**Sample Models:**")
                    for model in sample_models[:3]:  # Show first 3 models
                        output_lines.append(f"  • `{model}`")

                    if len(sample_models) > 3:
                        output_lines.append(
                            f"  • ... and {len(sample_models) - 3} more"
                        )

                output_lines.append("")  # Empty line between engines

            # Usage examples
            output_lines.append("\\n**Usage Examples:**")
            output_lines.append("```")
            output_lines.append(
                "/swap(meta-llama/Llama-3.1-8B-Instruct)     # Local model"
            )
            output_lines.append(
                "/swap(anthropic:claude-3-5-sonnet-20241022)  # API model"
            )
            output_lines.append(
                "/swap(config:path/to/config.yaml)            # Config-based swap"
            )
            output_lines.append("```")

            content = "\\n".join(output_lines)

            # Display in a panel
            panel = Panel(
                content,
                title="🔧 Inference Engines" if use_emoji else "Inference Engines",
                border_style=getattr(self._style, "assistant_border_style", "cyan"),
            )
            self.console.print(panel)

            return CommandResult(success=True, should_continue=False)

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error listing engines: {str(e)}",
                should_continue=False,
            )

    def _get_engines_info(self) -> list[dict]:
        """Get information about available inference engines."""
        engines = [
            {
                "name": "NATIVE",
                "type": "Local",
                "description": "PyTorch/transformers inference for smaller models",
                "sample_models": [
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "microsoft/Phi-3.5-mini-instruct",
                    "Qwen/Qwen2.5-3B-Instruct",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                ],
                "api_key_required": False,
            },
            {
                "name": "VLLM",
                "type": "Local",
                "description": "High-performance local inference for large models",
                "sample_models": [
                    "meta-llama/Llama-3.1-70B-Instruct",
                    "meta-llama/Llama-3.1-405B-Instruct",
                    "Qwen/Qwen2.5-72B-Instruct",
                    "microsoft/Phi-4-14B",
                ],
                "api_key_required": False,
            },
            {
                "name": "LLAMACPP",
                "type": "Local",
                "description": "CPU/GPU optimized inference with GGUF quantized models",
                "sample_models": [
                    "microsoft/Phi-3.5-mini-instruct (GGUF)",
                    "meta-llama/Llama-3.1-8B-Instruct (GGUF)",
                    "Qwen/Qwen2.5-7B-Instruct (GGUF)",
                ],
                "api_key_required": False,
            },
            {
                "name": "SGLANG",
                "type": "Local",
                "description": "Complex generation patterns and vision model support",
                "sample_models": [
                    "meta-llama/Llama-3.2-11B-Vision-Instruct",
                    "Qwen/Qwen2-VL-7B-Instruct",
                    "microsoft/Phi-3.5-vision-instruct",
                ],
                "api_key_required": False,
            },
            {
                "name": "ANTHROPIC",
                "type": "API",
                "description": "Claude models via Anthropic API",
                "sample_models": [
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                ],
                "api_key_required": True,
            },
            {
                "name": "OPENAI",
                "type": "API",
                "description": "GPT models via OpenAI API",
                "sample_models": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-3.5-turbo",
                    "o1-preview",
                ],
                "api_key_required": True,
            },
            {
                "name": "TOGETHER",
                "type": "API",
                "description": "Large model catalog via Together AI",
                "sample_models": [
                    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    "meta-llama/Llama-4-Maverick-32B-16E-Instruct",
                    "deepseek-ai/DeepSeek-R1",
                    "Qwen/QwQ-32B-Preview",
                ],
                "api_key_required": True,
            },
            {
                "name": "DEEPSEEK",
                "type": "API",
                "description": "DeepSeek models via DeepSeek API",
                "sample_models": [
                    "deepseek-chat",
                    "deepseek-reasoner",
                    "deepseek-coder",
                    "deepseek-math",
                ],
                "api_key_required": True,
            },
            {
                "name": "GOOGLE_VERTEX",
                "type": "API",
                "description": "Google models via Vertex AI",
                "sample_models": [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "text-bison@002",
                ],
                "api_key_required": True,
            },
            {
                "name": "GEMINI",
                "type": "API",
                "description": "Google Gemini models via direct API",
                "sample_models": [
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "gemini-1.0-pro",
                ],
                "api_key_required": True,
            },
            {
                "name": "REMOTE_VLLM",
                "type": "Remote",
                "description": "Connect to external vLLM server instances",
                "sample_models": ["Custom models hosted on remote vLLM servers"],
                "api_key_required": False,
            },
            {
                "name": "REMOTE",
                "type": "Remote",
                "description": "Generic OpenAI-compatible API endpoints",
                "sample_models": ["Models from OpenAI-compatible services"],
                "api_key_required": False,
            },
        ]

        return engines

    def _save_current_model_state_to_branch(self, branch_id: str):
        """Save current model configuration to a branch."""
        try:
            if hasattr(self.context, "branch_manager") and self.context.branch_manager:
                branch = self.context.branch_manager.get_branch_by_id(branch_id)
                if branch:
                    # Save model and generation configs
                    branch["model_config"] = self._serialize_model_config(
                        self.config.model
                    )
                    branch["generation_config"] = self._serialize_generation_config(
                        self.config.generation
                    )
        except Exception:
            # Silently fail to avoid disrupting user experience
            pass

    def _restore_model_state_from_branch(self, branch: dict[str, Any]):
        """Restore model configuration from a branch."""
        try:
            # This would require infrastructure changes to actually swap models
            # For now, just store the state
            if "model_config" in branch or "generation_config" in branch:
                pass  # Placeholder for actual restoration logic
        except Exception:
            # Silently fail to avoid disrupting user experience
            pass

    def _serialize_model_config(self, model_config) -> dict:
        """Serialize model config to dictionary."""
        if model_config is None:
            return {}

        config_dict = {}
        for attr in [
            "model_name",
            "model_max_length",
            "torch_dtype_str",
            "attn_implementation",
        ]:
            if hasattr(model_config, attr):
                value = getattr(model_config, attr)
                if value is not None:
                    config_dict[attr] = str(value)

        return config_dict

    def _serialize_generation_config(self, generation_config) -> dict:
        """Serialize generation config to dictionary."""
        if generation_config is None:
            return {}

        config_dict = {}
        for attr in ["max_new_tokens", "temperature", "top_p", "top_k", "sampling"]:
            if hasattr(generation_config, attr):
                value = getattr(generation_config, attr)
                if value is not None:
                    config_dict[attr] = value

        return config_dict
