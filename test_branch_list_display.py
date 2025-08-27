#!/usr/bin/env python3
"""
Quick test script to verify that the branch list API is working correctly.
This will help identify if the GUI branch list display issue is resolved.
"""

import json
import logging
import requests
import time
from oumi.core.configs import InferenceConfig
from oumi.webchat.server import WebChatSession
from tests.utils.chat_test_utils import create_test_inference_config

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_branch_list_api():
    """Test that the branch list API returns all active branches correctly."""
    print("🧪 Testing branch list API functionality...")
    
    # Create a WebChat session with a real config
    config = create_test_inference_config()
    session = WebChatSession(session_id="branch_list_test", config=config)
    
    print(f"✅ Created session with ID: {session.session_id}")
    
    # Add some conversation history to the session
    test_messages = [
        {"role": "user", "content": "Hello, let's test branching!"},
        {"role": "assistant", "content": "Great! I'm ready to help test the branch functionality."},
        {"role": "user", "content": "Can you explain what branches are?"},
        {"role": "assistant", "content": "Branches allow you to explore different conversation paths from any point."},
    ]
    
    for msg in test_messages:
        session.conversation_history.append({
            "role": msg["role"],
            "content": msg["content"], 
            "timestamp": time.time()
        })
    
    print(f"✅ Added {len(test_messages)} messages to conversation")
    
    # Test 1: Get initial branches (should have main branch only)
    print("\n🧪 Test 1: Get initial branch list")
    initial_branches = session.branch_manager.list_branches()
    print(f"📋 Initial branches: {len(initial_branches)}")
    
    for branch in initial_branches:
        print(f"  - {branch['id']}: '{branch['name']}' (messages: {branch['message_count']}, current: {branch['is_current']})")
    
    assert len(initial_branches) == 1, f"Expected 1 initial branch, got {len(initial_branches)}"
    assert initial_branches[0]['id'] == 'main', f"Expected main branch, got {initial_branches[0]['id']}"
    assert initial_branches[0]['message_count'] == 4, f"Expected 4 messages, got {initial_branches[0]['message_count']}"
    print("✅ Initial branch list correct")
    
    # Test 2: Create multiple branches
    print("\n🧪 Test 2: Create multiple branches")
    branch_configs = [
        {"name": "explanation_branch", "branch_point": 2},
        {"name": "deep_dive", "branch_point": 3}, 
        {"name": "alternative_approach", "branch_point": 1},
    ]
    
    created_branches = []
    for config in branch_configs:
        print(f"  Creating branch '{config['name']}' from point {config['branch_point']}...")
        success, message, new_branch = session.branch_manager.create_branch(
            from_branch_id="main",
            name=config['name'],
            branch_point=config['branch_point']
        )
        assert success, f"Failed to create branch {config['name']}: {message}"
        created_branches.append(new_branch)
        print(f"  ✅ Created branch '{new_branch.id}' with {len(new_branch.conversation_history)} messages")
    
    # Test 3: Verify all branches are listed
    print("\n🧪 Test 3: Verify all branches are listed")
    all_branches = session.branch_manager.list_branches()
    print(f"📋 Total branches after creation: {len(all_branches)}")
    
    expected_branch_count = 1 + len(branch_configs)  # main + created branches
    assert len(all_branches) == expected_branch_count, f"Expected {expected_branch_count} branches, got {len(all_branches)}"
    
    # Verify each branch is present and has correct data
    branch_ids = [b['id'] for b in all_branches]
    expected_ids = ['main'] + [cb.id for cb in created_branches]
    
    for expected_id in expected_ids:
        assert expected_id in branch_ids, f"Missing branch: {expected_id}"
    
    for branch in all_branches:
        print(f"  - {branch['id']}: '{branch['name']}' (messages: {branch['message_count']}, current: {branch['is_current']}, parent: {branch['parent']})")
    
    print("✅ All branches correctly listed")
    
    # Test 4: Test branch switching updates the list correctly
    print("\n🧪 Test 4: Test branch switching")
    target_branch_id = created_branches[0].id
    print(f"  Switching to branch '{target_branch_id}'...")
    
    success, message, switched_branch = session.branch_manager.switch_branch(target_branch_id)
    assert success, f"Failed to switch to branch {target_branch_id}: {message}"
    print(f"  ✅ Successfully switched to '{switched_branch.name}'")
    
    # Verify the current branch is updated in the list
    updated_branches = session.branch_manager.list_branches()
    current_branches = [b for b in updated_branches if b['is_current']]
    assert len(current_branches) == 1, f"Expected exactly 1 current branch, got {len(current_branches)}"
    assert current_branches[0]['id'] == target_branch_id, f"Expected current branch {target_branch_id}, got {current_branches[0]['id']}"
    print("✅ Branch switching updates list correctly")
    
    # Test 5: Test branch deletion updates the list
    print("\n🧪 Test 5: Test branch deletion")
    branch_to_delete = created_branches[-1].id
    print(f"  Deleting branch '{branch_to_delete}'...")
    
    success, message = session.branch_manager.delete_branch(branch_to_delete)
    assert success, f"Failed to delete branch {branch_to_delete}: {message}"
    print(f"  ✅ Successfully deleted branch")
    
    # Verify branch is removed from list
    final_branches = session.branch_manager.list_branches()
    final_branch_ids = [b['id'] for b in final_branches]
    assert branch_to_delete not in final_branch_ids, f"Deleted branch {branch_to_delete} still in list"
    
    expected_final_count = expected_branch_count - 1  # One branch deleted
    assert len(final_branches) == expected_final_count, f"Expected {expected_final_count} branches after deletion, got {len(final_branches)}"
    print("✅ Branch deletion updates list correctly")
    
    # Final summary
    print(f"\n📋 Final branch state:")
    for branch in final_branches:
        print(f"  - {branch['id']}: '{branch['name']}' (messages: {branch['message_count']}, current: {branch['is_current']})")
    
    print("\n🎉 All branch list API tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_branch_list_api()
        print("✅ Branch list display test completed successfully!")
    except Exception as e:
        print(f"❌ Branch list display test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)