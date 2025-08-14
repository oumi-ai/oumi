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

"""Shared context object for command handlers."""

from typing import TYPE_CHECKING, Optional

from rich.console import Console

if TYPE_CHECKING:
    from oumi.core.attachments import ContextWindowManager, FileHandler
    from oumi.core.commands.compaction_engine import CompactionEngine
    from oumi.core.commands.conversation_branches import ConversationBranchManager
    from oumi.core.configs import InferenceConfig
    from oumi.core.inference import BaseInferenceEngine
    from oumi.core.thinking import ThinkingProcessor


class CommandContext:
    """Shared context for all command handlers.

    This class provides a centralized way to access common dependencies
    and state that all command handlers need.
    """

    def __init__(
        self,
        console: Console,
        config: "InferenceConfig",
        conversation_history: list,
        inference_engine: "BaseInferenceEngine",
        system_monitor=None,
    ):
        """Initialize the command context.

        Args:
            console: Rich console for output.
            config: Inference configuration.
            conversation_history: List of conversation messages.
            inference_engine: The inference engine being used.
            system_monitor: Optional system monitor for displaying stats.
        """
        self.console = console
        self.config = config
        self.conversation_history = conversation_history
        self.inference_engine = inference_engine
        self.system_monitor = system_monitor
        self._style = config.style

        # Lazy-initialized components
        self._context_window_manager: Optional[ContextWindowManager] = None
        self._file_handler: Optional[FileHandler] = None
        self._compaction_engine: Optional[CompactionEngine] = None
        self._branch_manager: Optional[ConversationBranchManager] = None
        self._thinking_processor: Optional[ThinkingProcessor] = None
        self._macro_manager = None
        self._command_router = None

    @property
    def context_window_manager(self) -> "ContextWindowManager":
        """Get or create the context window manager."""
        if self._context_window_manager is None:
            from oumi.core.attachments import ContextWindowManager

            # Try multiple possible attribute names for max context length
            max_context = None
            possible_context_attrs = [
                "model_max_length",  # Standard transformers
                "max_model_len",  # VLLM
                "max_tokens",  # Some configs
                "context_length",  # Alternative name
            ]

            model_config = getattr(self.config, "model", None)
            if model_config:
                for attr in possible_context_attrs:
                    if hasattr(model_config, attr):
                        max_context = getattr(model_config, attr, None)
                        if max_context:
                            break

            # Fallback to a reasonable default
            if not max_context:
                max_context = 4096

            self._context_window_manager = ContextWindowManager(
                model_name=getattr(model_config, "model_name", "default"),
                max_context_length=max_context,
            )
        return self._context_window_manager

    @property
    def file_handler(self) -> "FileHandler":
        """Get or create the file handler."""
        if self._file_handler is None:
            from oumi.core.attachments import FileHandler

            self._file_handler = FileHandler(
                context_manager=self.context_window_manager,
            )
        return self._file_handler

    @property
    def compaction_engine(self) -> "CompactionEngine":
        """Get or create the compaction engine."""
        if self._compaction_engine is None:
            from oumi.core.commands.compaction_engine import CompactionEngine

            self._compaction_engine = CompactionEngine(
                console=self.console,
                inference_engine=self.inference_engine,
                config=self.config,
            )
        return self._compaction_engine

    @property
    def branch_manager(self) -> "ConversationBranchManager":
        """Get or create the conversation branch manager."""
        if self._branch_manager is None:
            from oumi.core.commands.conversation_branches import (
                ConversationBranchManager,
            )

            self._branch_manager = ConversationBranchManager(
                conversation_history=self.conversation_history
            )
        return self._branch_manager

    @property
    def thinking_processor(self) -> "ThinkingProcessor":
        """Get or create the thinking processor."""
        if self._thinking_processor is None:
            from oumi.core.thinking import ThinkingProcessor

            self._thinking_processor = ThinkingProcessor()
        return self._thinking_processor

    @property
    def macro_manager(self):
        """Get or create the macro manager."""
        if self._macro_manager is None:
            try:
                from oumi.core.commands.macro_manager import MacroManager

                self._macro_manager = MacroManager()
            except ImportError:
                self._macro_manager = None
        return self._macro_manager

    def set_command_router(self, command_router):
        """Set the command router reference for auto-save functionality."""
        self._command_router = command_router
