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

"""Interactive branch tree visualization component."""

import json
from typing import Any, Dict, List, Optional

import gradio as gr


def create_branch_tree_component(
    session_id: str,
    server_url: str = "http://localhost:8000",
    initial_branches: Optional[list[dict[str, Any]]] = None,
):
    """Create an interactive branch tree visualization using D3.js.

    Args:
        session_id: Session ID for API calls.
        server_url: Backend server URL.
        initial_branches: Initial branch data (optional).

    Returns:
        Gradio HTML component with embedded D3.js tree.
    """
    # Default branch data if not provided
    if initial_branches is None:
        initial_branches = [
            {
                "id": "main",
                "name": "Main",
                "message_count": 0,
                "created_at": "2025-01-01T00:00:00",
                "preview": "Empty branch",
                "parent": None,
                "is_current": True,
            }
        ]

    # Create a simpler HTML-only component first to test if the issue is with scripts
    branch_tree_html = f"""
    <div id="branch-container-{session_id}" style="width: 100%; min-height: 400px; border: 2px solid #007bff; border-radius: 8px; padding: 16px; background: #f0f8ff;">
        <div id="branch-header-{session_id}" style="margin-bottom: 16px;">
            <h4 style="margin: 0; color: #333;">🌿 Branch Tree (Session: {session_id[:8]}...)</h4>
            <p style="margin: 5px 0; color: #666; font-size: 12px;">Server: {server_url}</p>
        </div>

        <div id="branch-tree-{session_id}" style="width: 100%; height: 300px; border: 2px solid #28a745; border-radius: 4px; background: white; padding: 10px;">
            <p style="color: #666;">🔧 Branch visualization container (ID: branch-tree-{session_id})</p>
            <p style="color: #666;">📋 Initial branches: {len(initial_branches)} found</p>
            <div style="margin-top: 20px;">
                {_create_simple_branch_buttons(initial_branches, session_id, server_url)}
            </div>
        </div>

        <div id="branch-info-{session_id}" style="margin-top: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #666;">
            <div id="branch-details-{session_id}">Simple HTML test - if you see this, the component is rendering correctly.</div>
        </div>
    </div>
    """
    
    # Return both HTML component and JavaScript component
    html_component = gr.HTML(branch_tree_html)
    
    # Add click handlers using Gradio's JavaScript support
    def setup_branch_clicks():
        return f"""
        function() {{
            console.log("🚨 Gradio JS function executing for session: {session_id}");
            console.log("🚨 Looking for container: branch-tree-{session_id}");
            
            const container = document.getElementById("branch-tree-{session_id}");
            console.log("🚨 Container found:", !!container);
            
            if (container) {{
                console.log("🚨 Container details:", container);
                
                // Add simple click handlers to test buttons
                const buttons = container.querySelectorAll('.simple-branch-btn');
                console.log("🚨 Found", buttons.length, "branch buttons");
                
                buttons.forEach(btn => {{
                    btn.addEventListener('click', function(e) {{
                        const branchId = this.dataset.branchId;
                        console.log("🚨 Branch button clicked:", branchId);
                        
                        // Try to switch branch via API call
                        fetch("{server_url}/v1/oumi/branches", {{
                            method: "POST",
                            headers: {{ "Content-Type": "application/json" }},
                            body: JSON.stringify({{
                                "session_id": "{session_id}",
                                "action": "switch", 
                                "branch_id": branchId
                            }})
                        }})
                        .then(response => response.json())
                        .then(data => {{
                            console.log("🔄 Branch switch response:", data);
                            if (data.success) {{
                                alert("✅ Switched to branch: " + branchId);
                                // Optionally trigger conversation update here
                            }} else {{
                                alert("❌ Failed to switch branch: " + (data.message || "Unknown error"));
                            }}
                        }})
                        .catch(error => {{
                            console.error("❌ Branch switch error:", error);
                            alert("❌ Error switching branch: " + error);
                        }});
                    }});
                }});
            }}
            
            return "Initialization complete";
        }}
        """
    
    # Create a hidden button to trigger the JavaScript
    js_trigger = gr.Button("Initialize Branch Tree", visible=False)
    js_trigger.click(
        fn=lambda: "JS executed",
        outputs=None,
        js=setup_branch_clicks()
    )
    
    return html_component, js_trigger


def _create_simple_branch_buttons(branches, session_id, server_url):
    """Create simple HTML buttons for each branch to test clicking."""
    if not branches:
        return "<p>No branches to display</p>"
    
    buttons_html = ""
    for branch in branches:
        branch_id = branch.get('id', 'unknown')
        branch_name = branch.get('name', 'Unknown')
        is_current = branch.get('is_current', False)
        
        button_style = (
            "background: #007bff; color: white; border: 2px solid #0056b3;" if is_current 
            else "background: #6c757d; color: white; border: 2px solid #495057;"
        )
        
        buttons_html += f"""
        <button class="simple-branch-btn" data-branch-id="{branch_id}" 
                style="margin: 5px; padding: 8px 12px; {button_style} border-radius: 4px; cursor: pointer; font-size: 12px;">
            {branch_name} ({branch_id}) {"- CURRENT" if is_current else ""}
        </button>
        """
    
    return buttons_html
