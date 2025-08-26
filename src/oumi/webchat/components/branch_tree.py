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
) -> gr.HTML:
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

    # Convert branches to JSON for JavaScript
    branches_json = json.dumps(initial_branches)

    # Generate the HTML with embedded D3.js visualization
    branch_tree_html = f"""
    <div id="branch-container-{session_id}" style="width: 100%; min-height: 400px; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background: #fafafa;">
        <div id="branch-header-{session_id}" style="margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #333;">🌿 Conversation Branches</h4>
            <div id="branch-controls-{session_id}" style="display: flex; gap: 8px;">
                <button id="new-branch-btn-{session_id}" class="btn-primary" style="padding: 4px 8px; font-size: 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    + New Branch
                </button>
                <button id="refresh-btn-{session_id}" class="btn-secondary" style="padding: 4px 8px; font-size: 12px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    🔄 Refresh
                </button>
            </div>
        </div>

        <div id="branch-tree-{session_id}" style="width: 100%; height: 300px; border: 1px solid #eee; border-radius: 4px; background: white; overflow: hidden;"></div>

        <div id="branch-info-{session_id}" style="margin-top: 12px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; color: #666; min-height: 40px;">
            <div id="branch-details-{session_id}">Click on a branch to see details...</div>
        </div>

        <!-- Context menu (initially hidden) -->
        <div id="context-menu-{session_id}" style="display: none; position: absolute; background: white; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 1000; min-width: 150px;">
            <div class="context-menu-item" data-action="switch" style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #eee;">Switch to Branch</div>
            <div class="context-menu-item" data-action="branch" style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #eee;">Branch from Here</div>
            <div class="context-menu-item" data-action="rename" style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #eee;">Rename Branch</div>
            <div class="context-menu-item" data-action="delete" style="padding: 8px 12px; cursor: pointer; color: #dc3545;">Delete Branch</div>
        </div>
    </div>

    <!-- Load D3.js from CDN -->
    <script src="https://d3js.org/d3.v7.min.js"></script>

    <script>
    (function() {{
        // Unique identifiers for this instance
        const sessionId = "{session_id}";
        const containerId = `#branch-tree-${{sessionId}}`;
        const serverUrl = "{server_url}";

        // Initial branch data
        let branchData = {branches_json};
        let currentBranch = branchData.find(b => b.is_current)?.id || "main";

        // Tree configuration
        const config = {{
            width: 400,
            height: 280,
            nodeRadius: 20,
            marginTop: 20,
            marginLeft: 40,
            marginRight: 40,
            marginBottom: 20
        }};

        class BranchTreeVisualization {{
            constructor() {{
                this.container = d3.select(containerId);
                this.svg = null;
                this.g = null;
                this.tree = null;
                this.selectedBranch = null;

                this.init();
                this.render();
                this.setupEventHandlers();
            }}

            init() {{
                // Clear any existing content
                this.container.selectAll("*").remove();

                // Create SVG
                this.svg = this.container
                    .append("svg")
                    .attr("width", "100%")
                    .attr("height", config.height)
                    .attr("viewBox", `0 0 ${{config.width}} ${{config.height}}`);

                // Create main group
                this.g = this.svg.append("g")
                    .attr("transform", `translate(${{config.marginLeft}},${{config.marginTop}})`);

                // Configure tree layout
                this.tree = d3.tree()
                    .size([
                        config.height - config.marginTop - config.marginBottom,
                        config.width - config.marginLeft - config.marginRight
                    ]);
            }}

            buildHierarchy(branches) {{
                // Convert flat branch list to hierarchical structure
                const branchMap = new Map();
                branches.forEach(branch => {{
                    branchMap.set(branch.id, {{
                        ...branch,
                        children: []
                    }});
                }});

                let root = null;
                branches.forEach(branch => {{
                    const node = branchMap.get(branch.id);
                    if (branch.parent && branchMap.has(branch.parent)) {{
                        const parent = branchMap.get(branch.parent);
                        parent.children.push(node);
                    }} else {{
                        // This is a root node
                        if (!root) root = node;
                    }}
                }});

                return root || branchMap.get("main") || branches[0];
            }}

            render() {{
                // Clear previous rendering
                this.g.selectAll("*").remove();

                if (!branchData || branchData.length === 0) {{
                    this.showEmptyState();
                    return;
                }}

                // Build hierarchy
                const hierarchyRoot = this.buildHierarchy(branchData);
                if (!hierarchyRoot) {{
                    this.showEmptyState();
                    return;
                }}

                const root = d3.hierarchy(hierarchyRoot);
                const treeData = this.tree(root);

                // Draw links
                this.g.selectAll(".link")
                    .data(treeData.links())
                    .enter().append("path")
                    .attr("class", "link")
                    .attr("d", d3.linkHorizontal()
                        .x(d => d.y)
                        .y(d => d.x))
                    .style("fill", "none")
                    .style("stroke", "#999")
                    .style("stroke-width", "2px")
                    .style("stroke-dasharray", d => {{
                        // Dashed line for non-main branches
                        return d.target.data.id === "main" ? "none" : "3,3";
                    }});

                // Create node groups
                const node = this.g.selectAll(".node")
                    .data(treeData.descendants())
                    .enter().append("g")
                    .attr("class", "node")
                    .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
                    .style("cursor", "pointer");

                // Add node circles
                node.append("circle")
                    .attr("r", config.nodeRadius)
                    .style("fill", d => {{
                        if (d.data.is_current) return "#007bff";
                        if (d.data.id === "main") return "#28a745";
                        return "#ffffff";
                    }})
                    .style("stroke", d => {{
                        if (d.data.is_current) return "#0056b3";
                        if (d.data.id === "main") return "#1e7e34";
                        return "#6c757d";
                    }})
                    .style("stroke-width", "2px")
                    .style("filter", d => d.data.is_current ? "drop-shadow(0 0 4px rgba(0,123,255,0.5))" : "none");

                // Add node labels
                node.append("text")
                    .attr("dy", ".35em")
                    .attr("x", d => d.children ? -(config.nodeRadius + 5) : config.nodeRadius + 5)
                    .style("text-anchor", d => d.children ? "end" : "start")
                    .style("font-size", "11px")
                    .style("font-weight", d => d.data.is_current ? "bold" : "normal")
                    .style("fill", d => d.data.is_current ? "#007bff" : "#333")
                    .text(d => {{
                        const name = d.data.name || "Unknown";
                        return name.length > 10 ? name.substring(0, 8) + "..." : name;
                    }});

                // Add message count badges
                node.append("text")
                    .attr("dy", ".35em")
                    .attr("text-anchor", "middle")
                    .style("font-size", "9px")
                    .style("font-weight", "bold")
                    .style("fill", d => d.data.is_current ? "white" : "#666")
                    .style("pointer-events", "none")
                    .text(d => d.data.message_count || 0);

                // Add event handlers
                this.addNodeEventHandlers(node);
            }}

            addNodeEventHandlers(node) {{
                const self = this;

                // Click handler
                node.on("click", function(event, d) {{
                    event.stopPropagation();
                    self.handleNodeClick(d);
                }});

                // Right-click context menu
                node.on("contextmenu", function(event, d) {{
                    event.preventDefault();
                    event.stopPropagation();
                    self.showContextMenu(event, d);
                }});

                // Hover effects
                node.on("mouseover", function(event, d) {{
                    self.showBranchPreview(d.data);
                    // Highlight node
                    d3.select(this).select("circle")
                        .transition().duration(200)
                        .attr("r", config.nodeRadius + 3);
                }});

                node.on("mouseout", function(event, d) {{
                    // Reset node size
                    d3.select(this).select("circle")
                        .transition().duration(200)
                        .attr("r", config.nodeRadius);
                }});
            }}

            handleNodeClick(d) {{
                this.selectedBranch = d.data;
                this.showBranchPreview(d.data);

                // If not current branch, ask to switch
                if (!d.data.is_current) {{
                    this.switchToBranch(d.data.id, d.data.name);
                }}
            }}

            showBranchPreview(branchData) {{
                const detailsDiv = document.getElementById(`branch-details-${{sessionId}}`);
                if (detailsDiv) {{
                    const createdDate = new Date(branchData.created_at).toLocaleDateString();
                    const preview = branchData.preview || "No preview available";
                    const truncatedPreview = preview.length > 80 ? preview.substring(0, 77) + "..." : preview;

                    detailsDiv.innerHTML = `
                        <div><strong>${{branchData.name}}</strong> ${{branchData.is_current ? '(current)' : ''}}</div>
                        <div>Messages: ${{branchData.message_count}} • Created: ${{createdDate}}</div>
                        <div style="margin-top: 4px; font-style: italic;">${{truncatedPreview}}</div>
                    `;
                }}
            }}

            showContextMenu(event, d) {{
                const menu = document.getElementById(`context-menu-${{sessionId}}`);
                if (!menu) return;

                // Position menu
                menu.style.left = event.pageX + "px";
                menu.style.top = event.pageY + "px";
                menu.style.display = "block";

                // Store selected branch data
                menu.dataset.branchId = d.data.id;
                menu.dataset.branchName = d.data.name;

                // Disable delete for main branch
                const deleteItem = menu.querySelector('[data-action="delete"]');
                if (deleteItem) {{
                    deleteItem.style.opacity = d.data.id === "main" ? "0.5" : "1";
                    deleteItem.style.pointerEvents = d.data.id === "main" ? "none" : "auto";
                }}
            }}

            hideContextMenu() {{
                const menu = document.getElementById(`context-menu-${{sessionId}}`);
                if (menu) {{
                    menu.style.display = "none";
                }}
            }}

            showEmptyState() {{
                this.g.append("text")
                    .attr("x", config.width / 2 - config.marginLeft)
                    .attr("y", config.height / 2 - config.marginTop)
                    .attr("text-anchor", "middle")
                    .style("font-size", "14px")
                    .style("fill", "#999")
                    .text("No branches available");
            }}

            // API interaction methods
            async switchToBranch(branchId, branchName) {{
                try {{
                    const response = await fetch(`${{serverUrl}}/v1/oumi/branches`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            action: 'switch',
                            branch_id: branchId
                        }})
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        if (result.success) {{
                            // Update current branch
                            branchData = branchData.map(b => ({{
                                ...b,
                                is_current: b.id === branchId
                            }}));
                            currentBranch = branchId;

                            // Re-render tree
                            this.render();

                            // Show success message
                            this.showMessage(`Switched to branch: ${{branchName}}`, 'success');

                            // Trigger conversation update in parent interface
                            if (window.gradio_api) {{
                                window.gradio_api.refresh_conversation();
                            }}
                        }} else {{
                            this.showMessage(`Failed to switch: ${{result.message}}`, 'error');
                        }}
                    }}
                }} catch (error) {{
                    console.error('Error switching branch:', error);
                    this.showMessage(`Error: ${{error.message}}`, 'error');
                }}
            }}

            async createBranch(fromBranchId, branchName) {{
                if (!branchName || branchName.trim() === '') {{
                    branchName = `branch_${{Date.now()}}`;
                }}

                try {{
                    const response = await fetch(`${{serverUrl}}/v1/oumi/branches`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            action: 'create',
                            from_branch: fromBranchId,
                            name: branchName
                        }})
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        if (result.success) {{
                            await this.refreshBranches();
                            this.showMessage(`Created branch: ${{branchName}}`, 'success');
                        }} else {{
                            this.showMessage(`Failed to create branch: ${{result.message}}`, 'error');
                        }}
                    }}
                }} catch (error) {{
                    console.error('Error creating branch:', error);
                    this.showMessage(`Error: ${{error.message}}`, 'error');
                }}
            }}

            async deleteBranch(branchId, branchName) {{
                if (branchId === "main") {{
                    this.showMessage("Cannot delete main branch", 'error');
                    return;
                }}

                if (!confirm(`Delete branch "${{branchName}}"?`)) {{
                    return;
                }}

                try {{
                    const response = await fetch(`${{serverUrl}}/v1/oumi/branches`, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            session_id: sessionId,
                            action: 'delete',
                            branch_id: branchId
                        }})
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        if (result.success) {{
                            await this.refreshBranches();
                            this.showMessage(`Deleted branch: ${{branchName}}`, 'success');
                        }} else {{
                            this.showMessage(`Failed to delete branch: ${{result.message}}`, 'error');
                        }}
                    }}
                }} catch (error) {{
                    console.error('Error deleting branch:', error);
                    this.showMessage(`Error: ${{error.message}}`, 'error');
                }}
            }}

            async refreshBranches() {{
                try {{
                    const response = await fetch(`${{serverUrl}}/v1/oumi/branches?session_id=${{sessionId}}`);
                    if (response.ok) {{
                        const result = await response.json();
                        branchData = result.branches || [];
                        currentBranch = result.current_branch || "main";
                        this.render();
                    }}
                }} catch (error) {{
                    console.error('Error refreshing branches:', error);
                }}
            }}

            showMessage(message, type = 'info') {{
                const detailsDiv = document.getElementById(`branch-details-${{sessionId}}`);
                if (detailsDiv) {{
                    const colors = {{
                        success: '#28a745',
                        error: '#dc3545',
                        info: '#007bff'
                    }};

                    detailsDiv.innerHTML = `<div style="color: ${{colors[type] || colors.info}};">${{message}}</div>`;

                    // Clear message after 3 seconds
                    setTimeout(() => {{
                        if (this.selectedBranch) {{
                            this.showBranchPreview(this.selectedBranch);
                        }} else {{
                            detailsDiv.innerHTML = "Click on a branch to see details...";
                        }}
                    }}, 3000);
                }}
            }}
        }}

        // Initialize the visualization
        let treeViz = null;

        // Wait for DOM to be ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initTree);
        }} else {{
            setTimeout(initTree, 100); // Small delay to ensure Gradio has rendered
        }}

        function initTree() {{
            // Check if container exists
            const container = document.querySelector(containerId);
            if (container && !treeViz) {{
                treeViz = new BranchTreeVisualization();
                setupControls();
            }}
        }}

        function setupControls() {{
            // New branch button
            const newBranchBtn = document.getElementById(`new-branch-btn-${{sessionId}}`);
            if (newBranchBtn) {{
                newBranchBtn.onclick = () => {{
                    const branchName = prompt("Enter new branch name:");
                    if (branchName) {{
                        treeViz.createBranch(currentBranch, branchName);
                    }}
                }};
            }}

            // Refresh button
            const refreshBtn = document.getElementById(`refresh-btn-${{sessionId}}`);
            if (refreshBtn) {{
                refreshBtn.onclick = () => {{
                    if (treeViz) {{
                        treeViz.refreshBranches();
                    }}
                }};
            }}

            // Context menu handlers
            const contextMenu = document.getElementById(`context-menu-${{sessionId}}`);
            if (contextMenu) {{
                // Hide context menu when clicking elsewhere
                document.addEventListener('click', (event) => {{
                    if (!contextMenu.contains(event.target)) {{
                        contextMenu.style.display = 'none';
                    }}
                }});

                // Handle context menu item clicks
                contextMenu.addEventListener('click', (event) => {{
                    const action = event.target.dataset.action;
                    const branchId = contextMenu.dataset.branchId;
                    const branchName = contextMenu.dataset.branchName;

                    if (action && branchId && treeViz) {{
                        switch (action) {{
                            case 'switch':
                                treeViz.switchToBranch(branchId, branchName);
                                break;
                            case 'branch':
                                const newName = prompt(`Create new branch from "${{branchName}}":`);
                                if (newName) {{
                                    treeViz.createBranch(branchId, newName);
                                }}
                                break;
                            case 'rename':
                                const newBranchName = prompt(`Rename "${{branchName}}" to:`);
                                if (newBranchName) {{
                                    // TODO: Implement rename functionality
                                    alert("Rename functionality not implemented yet");
                                }}
                                break;
                            case 'delete':
                                treeViz.deleteBranch(branchId, branchName);
                                break;
                        }}
                    }}

                    contextMenu.style.display = 'none';
                }});

                // Style context menu items on hover
                contextMenu.querySelectorAll('.context-menu-item').forEach(item => {{
                    item.addEventListener('mouseover', () => {{
                        item.style.backgroundColor = '#f8f9fa';
                    }});
                    item.addEventListener('mouseout', () => {{
                        item.style.backgroundColor = 'white';
                    }});
                }});
            }}
        }}

        // Expose API for external communication
        window.branchTreeAPI_${{sessionId}} = {{
            refresh: () => treeViz?.refreshBranches(),
            switchTo: (branchId) => treeViz?.switchToBranch(branchId, branchId),
            createBranch: (name) => treeViz?.createBranch(currentBranch, name)
        }};

    }})();
    </script>
    """

    return gr.HTML(branch_tree_html)
