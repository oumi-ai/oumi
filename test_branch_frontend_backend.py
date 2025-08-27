#!/usr/bin/env python3
"""
Test the branch data format compatibility between frontend and backend.
This will help identify if the GUI display issue is due to data format mismatches.
"""

import json
import logging
import time
from oumi.webchat.server import WebChatSession
from tests.utils.chat_test_utils import create_test_inference_config

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_branch_data_format():
    """Test branch data format compatibility between frontend and backend."""
    print("🧪 Testing branch data format for frontend-backend compatibility...")
    
    # Create a WebChat session
    config = create_test_inference_config()
    session = WebChatSession(session_id="frontend_backend_test", config=config)
    
    # Add test conversation
    test_messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
    ]
    
    for msg in test_messages:
        session.conversation_history.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": time.time()
        })
    
    print(f"✅ Created session with {len(test_messages)} messages")
    
    # Test 1: Check initial branch data format
    print("\n🧪 Test 1: Initial branch data format")
    branches = session.branch_manager.list_branches()
    current_branch = session.branch_manager.current_branch_id
    
    # This simulates what the HTTP API returns
    api_response = {
        "branches": branches,
        "current_branch": current_branch
    }
    
    print("📋 API Response structure:")
    print(json.dumps(api_response, indent=2, default=str))
    
    # Check what the frontend expects
    assert "branches" in api_response, "Missing 'branches' in API response"
    assert "current_branch" in api_response, "Missing 'current_branch' in API response"
    assert isinstance(api_response["branches"], list), "branches should be a list"
    assert len(api_response["branches"]) > 0, "branches list should not be empty"
    
    # Check branch data structure
    main_branch = api_response["branches"][0]
    expected_fields = ["id", "name", "is_current", "message_count", "created_at", "preview", "parent"]
    
    print("\n🔍 Checking main branch structure:")
    for field in expected_fields:
        if field in main_branch:
            print(f"  ✅ {field}: {main_branch[field]}")
        else:
            print(f"  ❌ MISSING: {field}")
            
    for field in expected_fields:
        assert field in main_branch, f"Missing required field in branch data: {field}"
    
    print("✅ Initial branch data format correct")
    
    # Test 2: Create branches and check format
    print("\n🧪 Test 2: Multiple branches data format")
    
    # Create some branches
    success1, _, branch1 = session.branch_manager.create_branch("main", name="test_branch_1", branch_point=2)
    success2, _, branch2 = session.branch_manager.create_branch("main", name="test_branch_2", branch_point=3)
    
    assert success1 and success2, "Failed to create test branches"
    
    # Get updated branch list
    updated_branches = session.branch_manager.list_branches()
    updated_response = {
        "branches": updated_branches,
        "current_branch": session.branch_manager.current_branch_id
    }
    
    print("📋 Updated API Response with multiple branches:")
    print(json.dumps(updated_response, indent=2, default=str))
    
    # Verify we have all expected branches
    assert len(updated_response["branches"]) == 3, f"Expected 3 branches, got {len(updated_response['branches'])}"
    
    branch_ids = [b["id"] for b in updated_response["branches"]]
    expected_ids = ["main", branch1.id, branch2.id]
    for expected_id in expected_ids:
        assert expected_id in branch_ids, f"Missing branch ID: {expected_id}"
    
    print("✅ Multiple branches data format correct")
    
    # Test 3: Check if frontend can parse this data
    print("\n🧪 Test 3: Frontend data parsing simulation")
    
    # Simulate what the frontend JavaScript does
    def simulate_frontend_parsing(api_response):
        """Simulate frontend parsing of the API response."""
        try:
            branches = api_response.get("branches", [])
            current_branch = api_response.get("current_branch", "main")
            
            if not branches:
                return False, "No branches in response"
            
            # Check if we can find the current branch
            current_branch_found = False
            for branch in branches:
                if branch.get("is_current") == True:
                    current_branch_found = True
                    break
            
            if not current_branch_found:
                return False, "No current branch marked in branch list"
            
            # Check if all branches have required data for D3.js tree
            for i, branch in enumerate(branches):
                required = ["id", "name", "parent"]
                for field in required:
                    if field not in branch:
                        return False, f"Branch {i} missing required field: {field}"
            
            return True, f"Successfully parsed {len(branches)} branches"
            
        except Exception as e:
            return False, f"Parsing error: {str(e)}"
    
    # Test parsing
    parse_success, parse_message = simulate_frontend_parsing(updated_response)
    print(f"🖥️  Frontend parsing result: {parse_message}")
    
    assert parse_success, f"Frontend parsing failed: {parse_message}"
    print("✅ Frontend can successfully parse the data")
    
    # Test 4: Branch switching and data consistency
    print("\n🧪 Test 4: Branch switching data consistency")
    
    # Switch to a different branch
    target_branch_id = branch1.id
    success, message, switched_branch = session.branch_manager.switch_branch(target_branch_id)
    assert success, f"Failed to switch branch: {message}"
    
    # Get data after switching
    switched_response = {
        "branches": session.branch_manager.list_branches(),
        "current_branch": session.branch_manager.current_branch_id
    }
    
    print("📋 After branch switch:")
    print(json.dumps(switched_response, indent=2, default=str))
    
    # Verify the current_branch field matches the is_current flags
    current_from_field = switched_response["current_branch"]
    current_from_flags = [b["id"] for b in switched_response["branches"] if b["is_current"]]
    
    assert current_from_field == target_branch_id, f"current_branch field incorrect: {current_from_field} != {target_branch_id}"
    assert len(current_from_flags) == 1, f"Expected exactly 1 current branch in flags, got {len(current_from_flags)}"
    assert current_from_flags[0] == target_branch_id, f"is_current flag incorrect: {current_from_flags[0]} != {target_branch_id}"
    
    print("✅ Branch switching data consistency correct")
    
    # Test 5: Verify D3.js tree structure requirements
    print("\n🧪 Test 5: D3.js tree structure requirements")
    
    def simulate_d3_hierarchy_build(branches):
        """Simulate the D3.js hierarchy building logic from the frontend."""
        try:
            # This mirrors the buildHierarchy() function in the frontend
            branch_map = {}
            for branch in branches:
                branch_map[branch["id"]] = {
                    **branch,
                    "children": []
                }
            
            root = None
            for branch in branches:
                node = branch_map[branch["id"]]
                parent_id = branch.get("parent")
                if parent_id and parent_id in branch_map:
                    parent = branch_map[parent_id]
                    parent["children"].append(node)
                else:
                    # This is a root node
                    if not root:
                        root = node
            
            # If no root found, use main or first branch
            if not root:
                root = branch_map.get("main") or list(branch_map.values())[0]
            
            return True, root, f"Built tree with {len(branches)} nodes"
            
        except Exception as e:
            return False, None, f"Tree building error: {str(e)}"
    
    tree_success, tree_root, tree_message = simulate_d3_hierarchy_build(switched_response["branches"])
    print(f"🌳 D3.js tree building result: {tree_message}")
    
    if tree_success:
        def print_tree(node, indent=0):
            """Print the tree structure."""
            spaces = "  " * indent
            print(f"{spaces}- {node['name']} ({node['id']}) - {len(node['children'])} children")
            for child in node["children"]:
                print_tree(child, indent + 1)
        
        print("🌳 Tree structure:")
        print_tree(tree_root)
    
    assert tree_success, f"D3.js tree building failed: {tree_message}"
    print("✅ D3.js tree structure requirements met")
    
    print("\n🎉 All frontend-backend compatibility tests passed!")
    print("📝 The branch data format is correct and compatible with the frontend.")
    print("📝 If the GUI is not showing all branches, the issue is likely in:")
    print("    1. Frontend API calls not being made")
    print("    2. Frontend not refreshing after operations") 
    print("    3. Frontend filtering/hiding branches")
    print("    4. CSS/HTML rendering issues")
    
    return True

def main():
    try:
        test_branch_data_format()
        print("✅ Branch frontend-backend compatibility test completed successfully!")
    except Exception as e:
        print(f"❌ Branch compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)