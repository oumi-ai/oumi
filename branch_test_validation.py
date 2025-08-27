#!/usr/bin/env python3
"""
Quick validation test for WebChat branch functionality fixes.

This script tests the key branch operations to ensure all fixes are working.
"""

import asyncio
import json
import requests
import time


async def test_branch_functionality():
    """Test all branch functionality with a live server."""
    server_url = "http://localhost:9000"
    session_id = "test_validation_session"
    
    print("🧪 Testing WebChat Branch Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Create a session and add some conversation
        print("\n1️⃣  Testing session creation and conversation...")
        chat_response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello, what is Python?"}],
                "session_id": session_id,
                "max_tokens": 50
            },
            timeout=10
        )
        
        if chat_response.status_code == 200:
            print("✅ Chat response successful")
        else:
            print(f"❌ Chat failed: {chat_response.status_code}")
            return
            
        # Test 2: List initial branches
        print("\n2️⃣  Testing initial branch list...")
        branches_response = requests.get(
            f"{server_url}/v1/oumi/branches?session_id={session_id}"
        )
        
        if branches_response.status_code == 200:
            data = branches_response.json()
            branches = data.get("branches", [])
            print(f"✅ Initial branches: {[b['id'] for b in branches]}")
            print(f"✅ Current branch: {data.get('current_branch')}")
        else:
            print(f"❌ Failed to get branches: {branches_response.status_code}")
            return
            
        # Test 3: Create a new branch
        print("\n3️⃣  Testing branch creation...")
        create_response = requests.post(
            f"{server_url}/v1/oumi/branches",
            json={
                "session_id": session_id,
                "action": "create",
                "name": "test_branch_validation",
                "from_branch": "main"
            }
        )
        
        if create_response.status_code == 200:
            data = create_response.json()
            if data.get("success"):
                print(f"✅ Branch created: {data.get('message')}")
                created_branches = data.get("branches", [])
                print(f"✅ Total branches now: {[b['id'] for b in created_branches]}")
            else:
                print(f"❌ Branch creation failed: {data.get('message')}")
                return
        else:
            print(f"❌ Branch creation HTTP error: {create_response.status_code}")
            return
            
        # Test 4: Verify branch persistence
        print("\n4️⃣  Testing branch persistence...")
        time.sleep(0.5)  # Small delay
        
        verify_response = requests.get(
            f"{server_url}/v1/oumi/branches?session_id={session_id}"
        )
        
        if verify_response.status_code == 200:
            data = verify_response.json()
            branches = data.get("branches", [])
            branch_ids = [b['id'] for b in branches]
            
            if len(branches) >= 2 and any('test_branch_validation' in b['name'] for b in branches):
                print(f"✅ Branch persistence verified: {branch_ids}")
                print("✅ Created branch found in list!")
                
                # Find the created branch
                test_branch = None
                for b in branches:
                    if 'test_branch_validation' in b['name']:
                        test_branch = b
                        break
                
                if test_branch:
                    print(f"✅ Test branch details: {test_branch['name']} ({test_branch['id']})")
                    
                    # Test 5: Switch to the new branch
                    print("\n5️⃣  Testing branch switching...")
                    switch_response = requests.post(
                        f"{server_url}/v1/oumi/branches",
                        json={
                            "session_id": session_id,
                            "action": "switch",
                            "branch_id": test_branch['id']
                        }
                    )
                    
                    if switch_response.status_code == 200:
                        data = switch_response.json()
                        if data.get("success"):
                            print(f"✅ Branch switch successful: {data.get('message')}")
                            print(f"✅ Current branch now: {data.get('current_branch')}")
                        else:
                            print(f"❌ Branch switch failed: {data.get('message')}")
                    else:
                        print(f"❌ Branch switch HTTP error: {switch_response.status_code}")
                        
            else:
                print(f"❌ Branch persistence failed: {branch_ids}")
                print("❌ Created branch not found in subsequent list!")
                
        else:
            print(f"❌ Failed to verify branches: {verify_response.status_code}")
            
        print("\n🎉 Branch functionality test complete!")
        print("If you see ✅ for all tests, the branch storage bug is fixed!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the WebChat server is running on http://localhost:9000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(test_branch_functionality())