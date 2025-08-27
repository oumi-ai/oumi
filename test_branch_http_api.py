#!/usr/bin/env python3
"""
Test the actual HTTP API endpoint for branch listing to identify GUI display issues.
"""

import asyncio
import json
import logging
import time
from aiohttp import web
from oumi.webchat.server import OumiWebServer

# Enable debug logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_branch_http_api():
    """Test the HTTP API endpoints for branch operations."""
    print("🧪 Testing WebChat HTTP API for branch functionality...")
    
    # Create server instance
    server = OumiWebServer(port=8765)  # Use different port to avoid conflicts
    
    try:
        # Start the server
        print("📡 Starting WebChat server...")
        await server.start()
        print(f"✅ Server started on port {server.port}")
        
        # Create a session by calling the server's session management
        session_id = "test_http_api_session"
        test_session = await server.get_or_create_session(session_id)
        print(f"✅ Created session: {session_id}")
        
        # Add some conversation messages to the session
        test_messages = [
            {"role": "user", "content": "Hello, testing HTTP API!"},
            {"role": "assistant", "content": "Great! Let's test the branch HTTP endpoints."},
            {"role": "user", "content": "Please create multiple branches."},
            {"role": "assistant", "content": "I'll help you test branch creation via the API."},
        ]
        
        for msg in test_messages:
            test_session.conversation_history.append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": time.time()
            })
        
        print(f"✅ Added {len(test_messages)} messages to session")
        
        # Test 1: GET branches endpoint (initial state)
        print("\n🧪 Test 1: GET /v1/oumi/branches (initial state)")
        
        # Create a mock request object
        class MockRequest:
            def __init__(self, method="GET", query=None):
                self.method = method
                self.query = query or {}
                
            async def json(self):
                return {}
        
        get_request = MockRequest("GET", {"session_id": session_id})
        response = await server.handle_branches_api(get_request)
        
        # Parse the response
        response_data = json.loads(response.text)
        print(f"📋 GET response: {json.dumps(response_data, indent=2)}")
        
        assert "branches" in response_data, "Response missing 'branches' field"
        assert "current_branch" in response_data, "Response missing 'current_branch' field"
        assert len(response_data["branches"]) == 1, f"Expected 1 branch, got {len(response_data['branches'])}"
        assert response_data["current_branch"] == "main", f"Expected current_branch='main', got {response_data['current_branch']}"
        
        # Verify branch structure
        main_branch = response_data["branches"][0]
        required_fields = ["id", "name", "is_current", "message_count", "preview", "created_at", "parent"]
        for field in required_fields:
            assert field in main_branch, f"Branch missing required field: {field}"
        
        assert main_branch["id"] == "main", f"Expected main branch id='main', got {main_branch['id']}"
        assert main_branch["message_count"] == 4, f"Expected 4 messages, got {main_branch['message_count']}"
        assert main_branch["is_current"] == True, f"Expected main branch to be current"
        
        print("✅ Initial GET branches API working correctly")
        
        # Test 2: POST create branches
        print("\n🧪 Test 2: POST /v1/oumi/branches (create branches)")
        
        branch_create_data = [
            {"action": "create", "name": "feature_branch", "from_branch": "main"},
            {"action": "create", "name": "experiment", "from_branch": "main"}, 
            {"action": "create", "name": "bugfix", "from_branch": "main"},
        ]
        
        created_branch_ids = []
        
        for create_data in branch_create_data:
            create_data["session_id"] = session_id
            
            class MockCreateRequest(MockRequest):
                def __init__(self, data):
                    super().__init__("POST")
                    self._data = data
                    self.query = {"session_id": session_id}
                    
                async def json(self):
                    return self._data
            
            create_request = MockCreateRequest(create_data)
            create_response = await server.handle_branches_api(create_request)
            create_result = json.loads(create_response.text)
            
            print(f"  📝 Creating '{create_data['name']}'...")
            print(f"     Response: {create_result}")
            
            assert create_result.get("success") == True, f"Failed to create branch {create_data['name']}: {create_result.get('message')}"
            
        print("✅ Branch creation via POST API working")
        
        # Test 3: GET branches after creation (should show all branches)
        print("\n🧪 Test 3: GET /v1/oumi/branches (after creation)")
        
        get_all_request = MockRequest("GET", {"session_id": session_id})
        get_all_response = await server.handle_branches_api(get_all_request)
        all_branches_data = json.loads(get_all_response.text)
        
        print(f"📋 GET all branches response:")
        print(json.dumps(all_branches_data, indent=2))
        
        assert "branches" in all_branches_data, "Response missing 'branches' field"
        expected_branch_count = 1 + len(branch_create_data)  # main + created branches
        actual_branch_count = len(all_branches_data["branches"])
        
        assert actual_branch_count == expected_branch_count, f"Expected {expected_branch_count} branches, got {actual_branch_count}"
        
        # Verify all branches have required data
        branch_names = [b["name"] for b in all_branches_data["branches"]]
        expected_names = ["Main", "feature_branch", "experiment", "bugfix"]
        
        for expected_name in expected_names:
            assert expected_name in branch_names, f"Missing branch: {expected_name}"
        
        # Print branch details
        print("📋 All branches found:")
        for branch in all_branches_data["branches"]:
            print(f"  - {branch['id']}: '{branch['name']}' (messages: {branch['message_count']}, current: {branch['is_current']})")
        
        print("✅ All branches correctly returned by GET API")
        
        # Test 4: Branch switching via POST
        print("\n🧪 Test 4: POST /v1/oumi/branches (switch branch)")
        
        # Find a non-main branch to switch to
        target_branch = None
        for branch in all_branches_data["branches"]:
            if branch["id"] != "main":
                target_branch = branch
                break
        
        assert target_branch is not None, "No non-main branch found to switch to"
        
        switch_data = {
            "session_id": session_id,
            "action": "switch",
            "branch_id": target_branch["id"]
        }
        
        switch_request = MockCreateRequest(switch_data)
        switch_response = await server.handle_branches_api(switch_request)
        switch_result = json.loads(switch_response.text)
        
        print(f"  🔀 Switching to '{target_branch['name']}' (ID: {target_branch['id']})")
        print(f"     Response: {switch_result}")
        
        assert switch_result.get("success") == True, f"Failed to switch branch: {switch_result.get('message')}"
        assert switch_result.get("current_branch") == target_branch["id"], f"Expected current branch {target_branch['id']}, got {switch_result.get('current_branch')}"
        
        print("✅ Branch switching via POST API working")
        
        # Test 5: GET branches after switch (verify current branch updated)
        print("\n🧪 Test 5: GET /v1/oumi/branches (after switch)")
        
        final_get_request = MockRequest("GET", {"session_id": session_id})
        final_response = await server.handle_branches_api(final_get_request)
        final_data = json.loads(final_response.text)
        
        print(f"📋 Final GET response:")
        for branch in final_data["branches"]:
            print(f"  - {branch['id']}: '{branch['name']}' (current: {branch['is_current']})")
        
        # Verify only one branch is current and it's the switched-to branch
        current_branches = [b for b in final_data["branches"] if b["is_current"]]
        assert len(current_branches) == 1, f"Expected exactly 1 current branch, got {len(current_branches)}"
        assert current_branches[0]["id"] == target_branch["id"], f"Expected current branch {target_branch['id']}, got {current_branches[0]['id']}"
        assert final_data["current_branch"] == target_branch["id"], f"Expected current_branch {target_branch['id']}, got {final_data['current_branch']}"
        
        print("✅ Current branch correctly updated after switch")
        
        print("\n🎉 All HTTP API tests passed! The backend is working correctly.")
        print("📝 This suggests the GUI issue is in the frontend, not the backend API.")
        
        return True
        
    finally:
        # Clean up
        print("🧹 Cleaning up server...")
        await server.stop()
        print("✅ Server stopped")

async def main():
    try:
        await test_branch_http_api()
        print("✅ Branch HTTP API test completed successfully!")
    except Exception as e:
        print(f"❌ Branch HTTP API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)