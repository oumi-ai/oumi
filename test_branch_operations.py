#!/usr/bin/env python3

import requests
import json
import sys

# Backend URL - adjust if needed
BACKEND_URL = "http://localhost:9000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint and return the response."""
    url = f"{BACKEND_URL}{endpoint}"
    headers = {'Content-Type': 'application/json'}
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        
        print(f"\n{method} {endpoint}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()
    except Exception as e:
        print(f"Error testing {endpoint}: {e}")
        return None

def main():
    """Test branch operations through API endpoints."""
    print("Testing Branch Operations API Endpoints")
    print("=" * 50)
    
    # Test 1: Get branches
    print("\n1. Getting current branches:")
    branches_response = test_endpoint("/v1/oumi/branches")
    
    # Test 2: Create a test branch
    print("\n2. Creating a test branch:")
    create_data = {
        "action": "create",
        "session_id": "default", 
        "name": "test_branch_api",
        "from_branch": "main"
    }
    create_response = test_endpoint("/v1/oumi/branches", "POST", create_data)
    
    # Test 3: Switch to the created branch
    print("\n3. Switching to test branch:")
    switch_data = {
        "command": "switch_branch",
        "args": ["test_branch_api"],
        "session_id": "default"
    }
    switch_response = test_endpoint("/v1/oumi/command", "POST", switch_data)
    
    # Test 4: Get branches again to see current state
    print("\n4. Getting branches after switch:")
    branches_after_switch = test_endpoint("/v1/oumi/branches")
    
    # Test 5: Switch back to main
    print("\n5. Switching back to main:")
    switch_main_data = {
        "command": "switch_branch", 
        "args": ["main"],
        "session_id": "default"
    }
    switch_main_response = test_endpoint("/v1/oumi/command", "POST", switch_main_data)
    
    # Test 6: Delete the test branch
    print("\n6. Deleting test branch:")
    delete_data = {
        "command": "delete_branch",
        "args": ["test_branch_api"],
        "session_id": "default"
    }
    delete_response = test_endpoint("/v1/oumi/command", "POST", delete_data)
    
    # Test 7: Final branch list
    print("\n7. Final branch list:")
    final_branches = test_endpoint("/v1/oumi/branches")
    
    print("\n" + "=" * 50)
    print("Branch operations testing complete!")

if __name__ == "__main__":
    main()