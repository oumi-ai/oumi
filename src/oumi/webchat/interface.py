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

"""Main Gradio interface for Oumi WebChat."""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import requests
from rich.console import Console

from oumi.core.configs import InferenceConfig
from oumi.webchat.components.branch_tree import create_branch_tree_component
from oumi.webchat.utils.gradio_helpers import format_conversation_for_gradio


class WebChatInterface:
    """Main WebChat interface using Gradio."""
    
    def __init__(self, config: InferenceConfig, server_url: str = "http://localhost:9000"):
        """Initialize the WebChat interface.
        
        Args:
            config: Inference configuration.
            server_url: URL of the Oumi WebChat server.
        """
        self.config = config
        self.server_url = server_url
        self.session_id = str(uuid.uuid4())
        self.console = Console()
        
        # API endpoints
        self.api_base = f"{server_url}/v1/oumi"
        self.command_endpoint = f"{self.api_base}/command"
        self.branches_endpoint = f"{self.api_base}/branches"
        self.chat_endpoint = f"{server_url}/v1/chat/completions"
        self.websocket_url = f"{server_url.replace('http', 'ws')}/v1/oumi/ws?session_id={self.session_id}"
        
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        
        # Custom CSS for better styling
        custom_css = """
        .branch-panel {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
        }
        
        .command-button {
            margin: 2px;
            padding: 4px 8px;
        }
        
        .system-monitor {
            font-family: monospace;
            background: #1e1e1e;
            color: #00ff00;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        .chat-container {
            max-height: 600px;
            overflow-y: auto;
        }
        
        .branch-tree-container {
            min-height: 400px;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 16px;
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            .gradio-container {
                flex-direction: column !important;
            }
            
            .branch-tree-container {
                min-height: 200px;
            }
        }
        """
        
        with gr.Blocks(
            title="Oumi WebChat", 
            css=custom_css,
            theme=gr.themes.Soft()
        ) as interface:
            
            # Header
            with gr.Row():
                gr.Markdown("# 🤖 Oumi WebChat")
                
                # Model info display
                model_info = gr.HTML(
                    value=self._get_model_info_html(),
                    elem_classes=["model-info"]
                )
            
            # Main layout - two columns
            with gr.Row():
                # Left column - Chat interface
                with gr.Column(scale=2):
                    # Chat display  
                    chatbot = gr.Chatbot(
                        value=[],
                        height=500,
                        show_copy_button=True,
                        show_share_button=False,
                        container=True,
                        type="messages",
                        elem_classes=["chat-container"],
                        label=self._get_chatbot_title()
                    )
                    
                    # Message input
                    with gr.Row():
                        message_input = gr.Textbox(
                            placeholder="Type your message or /command...",
                            scale=9,
                            show_label=False,
                            container=False
                        )
                        send_button = gr.Button("Send", variant="primary", scale=1)
                    
                    # Quick command buttons
                    with gr.Row():
                        clear_btn = gr.Button("Clear", size="sm", elem_classes=["command-button"])
                        delete_btn = gr.Button("Delete Last", size="sm", elem_classes=["command-button"]) 
                        regen_btn = gr.Button("Regenerate", size="sm", elem_classes=["command-button"])
                        
                    with gr.Row():
                        attach_btn = gr.UploadButton(
                            "Attach File", 
                            file_types=["image", ".pdf", ".txt", ".json", ".csv", ".md"],
                            file_count="multiple",
                            size="sm"
                        )
                        export_btn = gr.Button("Export", size="sm", elem_classes=["command-button"])
                        help_btn = gr.Button("Help", size="sm", elem_classes=["command-button"])
                        
                # Right column - Branch tree and controls
                with gr.Column(scale=1):
                    # Branch tree visualization
                    with gr.Accordion("Conversation Branches", open=True):
                        branch_tree = create_branch_tree_component(
                            session_id=self.session_id,
                            server_url=self.server_url
                        )
                        
                        # Branch controls
                        with gr.Row():
                            new_branch_btn = gr.Button("New Branch", size="sm", variant="secondary")
                            switch_branch_input = gr.Textbox(
                                placeholder="Branch name to switch...", 
                                scale=2,
                                show_label=False
                            )
                            switch_btn = gr.Button("Switch", size="sm")
                    
                    # System monitor
                    with gr.Accordion("System Monitor", open=False):
                        system_monitor = gr.HTML(
                            value=self._get_system_monitor_html(),
                            elem_classes=["system-monitor"]
                        )
                        refresh_monitor_btn = gr.Button("🔄 Refresh", size="sm")
                        
                    # Settings panel
                    with gr.Accordion("Settings", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=getattr(self.config.generation, 'temperature', 1.0),
                            step=0.1,
                            label="Temperature"
                        )
                        
                        max_tokens_slider = gr.Slider(
                            minimum=100,
                            maximum=4000,
                            value=getattr(self.config.generation, 'max_new_tokens', 2048),
                            step=100,
                            label="Max Tokens"
                        )
                        
                        model_selector = gr.Dropdown(
                            choices=self._get_available_models(),
                            value=getattr(self.config.model, 'model_name', 'Current Model'),
                            label="Model",
                            interactive=True
                        )
            
            # Hidden state components
            session_state = gr.State({"session_id": self.session_id, "conversation": []})
            current_branch = gr.State("main")
            branches_data = gr.State([])
            
            # Event handlers
            
            # Send message
            def handle_send_message(message: str, history: List, state: Dict) -> Tuple[List, str, Dict]:
                """Handle sending a chat message."""
                if not message.strip():
                    return history, "", state
                    
                # Add user message to history in messages format
                history.append({"role": "user", "content": message})
                
                # Check if it's a command
                if message.startswith('/'):
                    response = self._execute_command(message, state["session_id"])
                    if response.get("success"):
                        # Command executed successfully
                        if response.get("message"):
                            content = f"✅ {response['message']}"
                        else:
                            content = "✅ Command executed"
                    else:
                        content = f"❌ {response.get('message', 'Command failed')}"
                else:
                    # Regular chat message - send to inference engine
                    try:
                        # Prepare messages for OpenAI-compatible API
                        messages = []
                        for msg in state.get("conversation", []):
                            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        # Add the current user message
                        messages.append({"role": "user", "content": message})
                        
                        # Call chat completions API
                        import requests
                        response = requests.post(
                            self.chat_endpoint,
                            json={
                                "messages": messages,
                                "session_id": self.session_id,
                                "max_tokens": getattr(self.config.generation, 'max_new_tokens', 100),
                                "temperature": getattr(self.config.generation, 'temperature', 0.7),
                                "stream": False
                            },
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                content = result["choices"][0]["message"]["content"]
                            else:
                                content = "I couldn't generate a response."
                        else:
                            error_text = response.text if hasattr(response, 'text') else 'Unknown error'
                            content = f"Server Error ({response.status_code}): Backend may still be loading the model. Please wait a moment and try again."
                            
                    except Exception as e:
                        if "Connection refused" in str(e) or "Read timed out" in str(e):
                            content = "🔄 Backend is starting up. Please wait for the model to load and try again."
                        else:
                            content = f"Connection error: {str(e)}"
                
                # Add assistant response in messages format
                history.append({"role": "assistant", "content": content})
                
                # Update state
                state["conversation"] = history
                
                return history, "", state
            
            # Command execution
            def execute_command(command: str, state: Dict) -> Tuple[str, Dict]:
                """Execute a command and return result."""
                response = self._execute_command(command, state["session_id"])
                
                if response.get("success"):
                    message = f"✅ {response.get('message', 'Command executed')}"
                else:
                    message = f"❌ {response.get('message', 'Command failed')}"
                
                return message, state
            
            # Branch operations
            def create_new_branch(state: Dict) -> Tuple[gr.HTML, Dict]:
                """Create a new conversation branch."""
                branch_name = f"branch_{int(time.time())}"
                
                response = requests.post(
                    self.branches_endpoint,
                    json={
                        "session_id": state["session_id"],
                        "action": "create",
                        "name": branch_name,
                        "from_branch": state.get("current_branch", "main")
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        # Refresh branch tree
                        return self._update_branch_tree(state["session_id"]), state
                
                return gr.HTML("❌ Failed to create branch"), state
            
            def switch_branch(branch_name: str, state: Dict) -> Tuple[List, gr.HTML, Dict]:
                """Switch to a different branch."""
                if not branch_name:
                    return [], gr.HTML("❌ Please enter a branch name"), state
                    
                response = requests.post(
                    self.branches_endpoint,
                    json={
                        "session_id": state["session_id"],
                        "action": "switch", 
                        "branch_id": branch_name
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        # Update conversation with branch history
                        conversation = data.get("conversation", [])
                        history = format_conversation_for_gradio(conversation)
                        
                        state["conversation"] = history
                        state["current_branch"] = data.get("current_branch", branch_name)
                        
                        return history, self._update_branch_tree(state["session_id"]), state
                
                return [], gr.HTML("❌ Failed to switch branch"), state
            
            # File attachment
            def handle_file_attachment(files: List, state: Dict) -> Tuple[str, Dict]:
                """Handle file attachments."""
                if not files:
                    return "", state
                    
                # Execute attach command for each file
                messages = []
                for file in files:
                    response = self._execute_command(f'/attach("{file.name}")', state["session_id"])
                    if response.get("success"):
                        messages.append(f"✅ Attached {file.name}")
                    else:
                        messages.append(f"❌ Failed to attach {file.name}")
                
                # Return the attachment status as a message input placeholder
                attachment_summary = " | ".join(messages)
                return f"Files attached: {attachment_summary}", state
                
            # Wire up event handlers
            send_button.click(
                handle_send_message,
                inputs=[message_input, chatbot, session_state],
                outputs=[chatbot, message_input, session_state]
            )
            
            message_input.submit(
                handle_send_message,
                inputs=[message_input, chatbot, session_state], 
                outputs=[chatbot, message_input, session_state]
            )
            
            # Command buttons - execute commands and refresh chat
            def handle_clear_command(state):
                execute_command("/clear()", state)
                # Clear the chatbot display too
                return [], state
                
            def handle_delete_command(state):
                # First sync the conversation to the backend session
                self._sync_conversation_to_backend(state["session_id"], state.get("conversation", []))
                result, updated_state = execute_command("/delete()", state)
                # Get updated conversation from backend
                updated_conversation = self._get_conversation_from_backend(state["session_id"])
                updated_state["conversation"] = updated_conversation
                return updated_conversation, updated_state
                
            def handle_regen_command(state):
                # First sync the conversation to the backend session  
                self._sync_conversation_to_backend(state["session_id"], state.get("conversation", []))
                result, updated_state = execute_command("/regen()", state)
                
                # Check if regen command wants to regenerate (user_input_override)
                if "user_input_override" in str(result):
                    # Get the last user message from conversation
                    conversation = state.get("conversation", [])
                    if conversation:
                        # Find last user message
                        last_user_message = None
                        for msg in reversed(conversation):
                            if msg.get("role") == "user":
                                last_user_message = msg.get("content", "")
                                break
                        
                        if last_user_message:
                            # Remove last assistant response if present
                            if conversation and conversation[-1].get("role") == "assistant":
                                conversation.pop()
                            
                            # Regenerate response using chat API
                            try:
                                messages = []
                                for msg in conversation:
                                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                                        messages.append({"role": msg["role"], "content": msg["content"]})
                                
                                # Add the user message we're regenerating for
                                messages.append({"role": "user", "content": last_user_message})
                                
                                import requests
                                response = requests.post(
                                    self.chat_endpoint,
                                    json={
                                        "messages": messages,
                                        "session_id": state["session_id"],
                                        "max_tokens": getattr(self.config.generation, 'max_new_tokens', 100),
                                        "temperature": getattr(self.config.generation, 'temperature', 0.7),
                                        "stream": False
                                    },
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    result_data = response.json()
                                    if "choices" in result_data and len(result_data["choices"]) > 0:
                                        new_response = result_data["choices"][0]["message"]["content"]
                                        # Add regenerated response
                                        conversation.append({"role": "assistant", "content": new_response})
                                        updated_state["conversation"] = conversation
                                        return conversation, updated_state
                                        
                            except Exception as e:
                                # Add error message if regeneration failed
                                conversation.append({"role": "assistant", "content": f"Error regenerating: {str(e)}"})
                                updated_state["conversation"] = conversation
                                return conversation, updated_state
                
                # Fallback: get updated conversation from backend
                updated_conversation = self._get_conversation_from_backend(state["session_id"])
                updated_state["conversation"] = updated_conversation
                return updated_conversation, updated_state
                
            def handle_help_command(state):
                result, updated_state = execute_command("/help()", state)
                # Add help result as system message
                help_msg = {"role": "assistant", "content": f"**Help**: {result}"}
                conversation = updated_state.get("conversation", [])
                conversation.append(help_msg)
                updated_state["conversation"] = conversation
                return conversation, updated_state
                
            def handle_export_command(state):
                # Generate default filename with timestamp
                import time
                timestamp = int(time.time())
                filename = f"oumi_chat_{timestamp}.pdf"
                
                # Execute export command via backend
                result, updated_state = execute_command(f'/save({filename})', state)
                
                if result and "success" in str(result) and "failed" not in str(result):
                    export_msg = {"role": "assistant", "content": f"✅ **Exported**: {result}"}
                else:
                    export_msg = {"role": "assistant", "content": f"❌ **Export Failed**: {result}"}
                
                conversation = updated_state.get("conversation", [])
                conversation.append(export_msg)
                updated_state["conversation"] = conversation
                return conversation, updated_state
            
            clear_btn.click(
                handle_clear_command,
                inputs=[session_state],
                outputs=[chatbot, session_state]
            )
            
            delete_btn.click(
                handle_delete_command,
                inputs=[session_state],
                outputs=[chatbot, session_state]
            )
            
            regen_btn.click(
                handle_regen_command,
                inputs=[session_state],
                outputs=[chatbot, session_state]
            )
            
            help_btn.click(
                handle_help_command,
                inputs=[session_state],
                outputs=[chatbot, session_state]
            )
            
            export_btn.click(
                handle_export_command,
                inputs=[session_state],
                outputs=[chatbot, session_state]
            )
            
            # Branch operations
            new_branch_btn.click(
                create_new_branch,
                inputs=[session_state],
                outputs=[branch_tree, session_state]
            )
            
            switch_btn.click(
                switch_branch,
                inputs=[switch_branch_input, session_state],
                outputs=[chatbot, branch_tree, session_state]
            )
            
            # File attachment
            attach_btn.upload(
                handle_file_attachment,
                inputs=[attach_btn, session_state],
                outputs=[message_input, session_state]  # Show result in message input
            )
            
            # System monitor refresh
            def refresh_system_monitor():
                return self._get_system_monitor_html()
                
            refresh_monitor_btn.click(
                refresh_system_monitor,
                outputs=[system_monitor]
            )
            
            # Settings panel functionality
            def update_temperature(value, state):
                response = self._execute_command(f'/set(temperature={value})', state["session_id"])
                if response.get("success"):
                    return f"✅ Temperature updated to {value}", state
                else:
                    return f"❌ Failed to update temperature: {response.get('message', 'Unknown error')}", state
            
            def update_max_tokens(value, state):
                response = self._execute_command(f'/set(max_tokens={int(value)})', state["session_id"])
                if response.get("success"):
                    return f"✅ Max tokens updated to {int(value)}", state
                else:
                    return f"❌ Failed to update max tokens: {response.get('message', 'Unknown error')}", state
            
            def update_model(model_name, state):
                # Use swap command to change models
                response = self._execute_command(f'/swap({model_name})', state["session_id"])
                if response.get("success"):
                    # Update the config model name for title display
                    if hasattr(self.config, 'model') and hasattr(self.config.model, 'model_name'):
                        self.config.model.model_name = model_name
                    
                    # Create new chatbot with updated title
                    updated_chatbot = gr.Chatbot(
                        value=state.get("conversation", []),
                        height=500,
                        show_copy_button=True,
                        show_share_button=False,
                        container=True,
                        type="messages",
                        elem_classes=["chat-container"],
                        label=self._get_chatbot_title()
                    )
                    
                    # Also update the model info display
                    updated_model_info = gr.HTML(
                        value=self._get_model_info_html(),
                        elem_classes=["model-info"]
                    )
                    
                    return f"✅ Model switched to {model_name}", state, updated_chatbot, updated_model_info
                else:
                    # Return current chatbot unchanged if model switch failed
                    current_chatbot = gr.Chatbot(
                        value=state.get("conversation", []),
                        height=500,
                        show_copy_button=True,
                        show_share_button=False,
                        container=True,
                        type="messages",
                        elem_classes=["chat-container"],
                        label=self._get_chatbot_title()
                    )
                    current_model_info = gr.HTML(
                        value=self._get_model_info_html(),
                        elem_classes=["model-info"]
                    )
                    
                    return f"❌ Failed to switch model: {response.get('message', 'Unknown error')}", state, current_chatbot, current_model_info
            
            # Wire up settings event handlers
            temperature_slider.change(
                update_temperature,
                inputs=[temperature_slider, session_state],
                outputs=[message_input, session_state]
            )
            
            max_tokens_slider.change(
                update_max_tokens, 
                inputs=[max_tokens_slider, session_state],
                outputs=[message_input, session_state]
            )
            
            model_selector.change(
                update_model,
                inputs=[model_selector, session_state], 
                outputs=[message_input, session_state, chatbot, model_info]
            )
            
        return interface
    
    def _execute_command(self, command: str, session_id: str) -> Dict[str, Any]:
        """Execute a command via the API."""
        try:
            # Parse command
            if command.startswith('/'):
                command = command[1:]  # Remove leading slash
                
            # Split command and args
            if '(' in command and command.endswith(')'):
                cmd_name = command.split('(')[0]
                args_str = command[command.find('(')+1:-1]
                args = [arg.strip().strip('"\'') for arg in args_str.split(',') if arg.strip()]
            else:
                cmd_name = command
                args = []
            
            response = requests.post(
                self.command_endpoint,
                json={
                    "session_id": session_id,
                    "command": cmd_name,
                    "args": args
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "message": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": str(e)}
    
    def _get_model_info_html(self) -> str:
        """Get HTML for model information display."""
        model_name = getattr(self.config.model, 'model_name', 'Unknown Model')
        engine = str(self.config.engine) if hasattr(self.config, 'engine') else 'Unknown'
        
        return f"""
        <div style="text-align: right; padding: 8px; background: #f0f0f0; border-radius: 4px;">
            <strong>Model:</strong> {model_name}<br>
            <strong>Engine:</strong> {engine}<br>
            <strong>Session:</strong> {self.session_id[:8]}...
        </div>
        """
    
    def _get_chatbot_title(self) -> str:
        """Get formatted title for the chatbot component using Oumi chat pattern."""
        model_name = getattr(self.config.model, 'model_name', 'Assistant')
        
        # Clean up model name for display (same as Oumi chat)
        if "/" in model_name:
            display_model_name = model_name.split("/")[-1]  # Get last part after /
        else:
            display_model_name = model_name
            
        return display_model_name
    
    def _get_system_monitor_html(self) -> str:
        """Get HTML for system monitor display using backend SystemMonitor."""
        try:
            # Get system stats from backend
            response = requests.get(
                f"{self.server_url}/v1/oumi/system_stats",
                params={"session_id": self.session_id},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # GPU info
                if data.get('gpu_vram_used_gb') is not None:
                    gpu_info = f"{data['gpu_vram_used_gb']:.1f}GB / {data['gpu_vram_total_gb']:.1f}GB ({data['gpu_vram_percent']:.1f}%)"
                else:
                    gpu_info = "N/A"
                
                # Memory info
                ram_used_gb = data.get('ram_used_gb', 0)
                ram_total_gb = data.get('ram_total_gb', 0)
                ram_percent = data.get('ram_percent', 0)
                
                # Context info
                context_used = data.get('context_used_tokens', 0)
                context_max = data.get('context_max_tokens', 0)
                context_percent = data.get('context_percent', 0)
                context_info = f"{context_used}/{context_max} tokens ({context_percent:.1f}%)"
                
                return f"""
                <div class="system-monitor">
                    <strong>System Monitor</strong><br>
                    GPU: {gpu_info}<br>
                    Memory: {ram_used_gb:.1f}GB / {ram_total_gb:.1f}GB ({ram_percent:.1f}%)<br>
                    Context: {context_info}<br>
                    <br>
                    <small>Live stats from backend</small>
                </div>
                """
            else:
                # Fallback to basic info if backend unavailable
                return f"""
                <div class="system-monitor">
                    <strong>System Monitor</strong><br>
                    GPU: --<br>
                    Memory: --<br>
                    Context: --<br>
                    <br>
                    <small>Backend unavailable</small>
                </div>
                """
                
        except Exception as e:
            return f"""
            <div class="system-monitor">
                <strong>System Monitor</strong><br>
                GPU: --<br>
                Memory: --<br>
                Context: --<br>
                <br>
                <small>Error: {str(e)}</small>
            </div>
            """
    
    def _get_available_models(self) -> List[str]:
        """Get list of available models for the dropdown."""
        current_model = getattr(self.config.model, 'model_name', 'Current Model')
        
        # Common model options organized by category
        models = [
            current_model,
            "--- Local Models ---",
            "meta-llama/Llama-3.1-8B-Instruct",
            "microsoft/Qwen2.5-3B-Instruct", 
            "microsoft/Phi-3.5-mini-instruct",
            "--- API Models ---",
            "anthropic:claude-3-5-sonnet-20241022",
            "anthropic:claude-3-5-haiku-20241022",
            "openai:gpt-4o",
            "openai:gpt-4o-mini",
            "together:meta-llama/Llama-3.1-70B-Instruct-Turbo",
            "together:deepseek-ai/DeepSeek-R1",
            "--- Config Files ---",
            "config:configs/recipes/qwen2_5/inference/3b_infer.yaml",
            "config:configs/recipes/llama3_1/inference/8b_infer.yaml"
        ]
        
        return models
    
    def _sync_conversation_to_backend(self, session_id: str, conversation: list):
        """Sync conversation state from frontend to backend session."""
        try:
            # Send conversation to backend via a sync endpoint
            response = requests.post(
                f"{self.server_url}/v1/oumi/sync_conversation",
                json={
                    "session_id": session_id,
                    "conversation": conversation
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to sync conversation: {e}")
            return False

    def _get_conversation_from_backend(self, session_id: str) -> list:
        """Get current conversation state from backend session."""
        try:
            response = requests.get(
                f"{self.server_url}/v1/oumi/conversation",
                params={"session_id": session_id},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("conversation", [])
        except Exception as e:
            print(f"Failed to get conversation: {e}")
        return []

    def _update_branch_tree(self, session_id: str) -> gr.HTML:
        """Update the branch tree display."""
        try:
            response = requests.get(
                self.branches_endpoint,
                params={"session_id": session_id}
            )
            
            if response.status_code == 200:
                data = response.json()
                branches = data.get("branches", [])
                current_branch = data.get("current_branch", "main")
                
                # Generate HTML for branch list (simplified for now)
                branch_html = "<div class='branch-list'>"
                for branch in branches:
                    name = branch.get("name", "Unknown")
                    is_current = branch.get("id") == current_branch
                    style = "font-weight: bold; color: blue;" if is_current else ""
                    
                    branch_html += f"""
                    <div style="padding: 4px; {style}">
                        {'→ ' if is_current else '  '}{name}
                        <small>({branch.get('message_count', 0)} msgs)</small>
                    </div>
                    """
                branch_html += "</div>"
                
                return gr.HTML(branch_html)
        
        except Exception as e:
            return gr.HTML(f"Error loading branches: {e}")
        
        return gr.HTML("No branches loaded")


def create_webchat_interface(
    config: InferenceConfig, 
    server_url: str = "http://localhost:9000"
) -> gr.Blocks:
    """Create the WebChat interface.
    
    Args:
        config: Inference configuration.
        server_url: URL of the WebChat server.
        
    Returns:
        Gradio Blocks interface.
    """
    webchat = WebChatInterface(config, server_url)
    return webchat.create_interface()


def launch_webchat(
    config: InferenceConfig,
    server_url: str = "http://localhost:9000",
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860
):
    """Launch the WebChat interface.
    
    Args:
        config: Inference configuration.
        server_url: URL of the WebChat server.
        share: Whether to create a public link.
        server_name: Server hostname.
        server_port: Server port.
    """
    interface = create_webchat_interface(config, server_url)
    
    print(f"🌐 Launching Oumi WebChat interface...")
    print(f"📍 Web Interface: http://{server_name}:{server_port}")
    print(f"🔗 Backend Server: {server_url}")
    print(f"🛑 Press Ctrl+C to stop")
    
    interface.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True
    )