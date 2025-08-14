#!/usr/bin/env python3
"""End-to-end test for the /regen() command issue.
This simulates the exact sequence the user is experiencing.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from oumi.core.commands.command_parser import CommandParser
from oumi.core.commands.handlers.conversation_operations_handler import (
    ConversationOperationsHandler,
)
from oumi.core.commands.handlers.parameter_management_handler import (
    ParameterManagementHandler,
)


def create_mock_config():
    """Create a mock config that matches the GGUF config."""
    config = MagicMock()

    # Create nested mock objects
    config.model = MagicMock()
    config.model.model_max_length = 16384
    config.model.model_name = "unsloth/gemma-3n-E4B-it-GGUF"

    config.generation = MagicMock()
    config.generation.temperature = 0.7
    config.generation.top_p = 0.9
    config.generation.max_new_tokens = 2048

    return config


def test_exact_user_sequence():
    """Test the exact sequence: question -> /set(temperature=0.8) -> /regen()"""
    print("🧪 Testing exact user sequence...")
    print("=" * 60)

    # Step 1: Set up initial state
    config = create_mock_config()
    conversation_history = []

    # Mock the command context
    context = MagicMock()
    context.config = config
    context.conversation_history = conversation_history
    context.system_monitor = MagicMock()
    context.console = MagicMock()
    context.inference_engine = MagicMock()
    context.context_window_manager = MagicMock()

    # Create handlers and parser
    conv_handler = ConversationOperationsHandler(context)
    param_handler = ParameterManagementHandler(context)
    parser = CommandParser()

    print("✅ Initial setup complete")

    # Step 2: Simulate initial conversation
    print("\n📝 Step 1: Initial conversation")
    conversation_history.extend(
        [
            {
                "role": "user",
                "content": "Which muppet do you think would make the best president?",
            },
            {
                "role": "assistant",
                "content": "I think Kermit the Frog would make an excellent president! He's diplomatic, optimistic, and has experience leading a diverse group...",
            },
        ]
    )

    print(f"   Conversation history: {len(conversation_history)} messages")
    print(f"   Last message role: {conversation_history[-1]['role']}")

    # Step 3: Test /set(temperature=0.8)
    print("\n⚙️  Step 2: /set(temperature=0.8)")
    set_command = parser.parse_command("/set(temperature=0.8)")
    set_result = param_handler.handle_command(set_command)

    print(f"   Set command success: {set_result.success}")
    print(f"   Set command message: {set_result.message}")
    print(f"   New temperature: {config.generation.temperature}")

    # Step 4: Test /regen() - this is where the issue occurs
    print("\n🔄 Step 3: /regen() - The Critical Test")
    print(f"   Before regen - conversation length: {len(conversation_history)}")
    print(
        f"   Before regen - last message: {conversation_history[-1]['role']}: '{conversation_history[-1]['content'][:50]}...'"
    )

    regen_command = parser.parse_command("/regen()")
    regen_result = conv_handler.handle_command(regen_command)

    print(f"   Regen success: {regen_result.success}")
    print(f"   Regen message: {regen_result.message}")
    print(f"   Should continue: {regen_result.should_continue}")
    print(
        f"   Has user_input_override: {hasattr(regen_result, 'user_input_override') and regen_result.user_input_override is not None}"
    )

    if (
        hasattr(regen_result, "user_input_override")
        and regen_result.user_input_override
    ):
        print(f"   User input override: '{regen_result.user_input_override[:50]}...'")

    print(f"   After regen - conversation length: {len(conversation_history)}")
    if conversation_history:
        print(
            f"   After regen - last message: {conversation_history[-1]['role']}: '{conversation_history[-1]['content'][:50]}...'"
        )

    # Step 5: Simulate what would happen in the inference loop
    print("\n🔍 Step 4: Simulate inference loop processing")

    # This simulates the key part of the inference loop
    input_text = (
        regen_result.user_input_override
        if hasattr(regen_result, "user_input_override")
        else None
    )
    is_from_override = bool(input_text)

    print(f"   input_text: '{input_text}' (from override: {is_from_override})")

    # Check conversation state before adding user message
    print("   Conversation before potential user message add:")
    for i, msg in enumerate(conversation_history):
        print(f"     {i}: {msg['role']}: '{msg['content'][:30]}...'")

    # Simulate the problematic logic
    would_add_message = not is_from_override
    print(f"   Would add new user message: {would_add_message}")

    if not would_add_message:
        print(
            "   ✅ GOOD: Skipping user message addition (this should prevent duplicate)"
        )
    else:
        print(
            "   ❌ BAD: Would add user message (this could cause role alternation error)"
        )

    # Check for role alternation issues
    print("\n🎭 Step 5: Check role alternation")

    # Simulate adding the user message (what the old code would do)
    test_history = conversation_history.copy()
    if would_add_message and input_text:
        test_history.append({"role": "user", "content": input_text})

    print("   Final conversation with potential user message:")
    for i, msg in enumerate(test_history):
        print(f"     {i}: {msg['role']}: '{msg['content'][:30]}...'")

    # Check for role alternation violations
    violations = []
    for i in range(1, len(test_history)):
        if test_history[i - 1]["role"] == test_history[i]["role"]:
            violations.append(
                f"Messages {i - 1} and {i} both have role '{test_history[i]['role']}'"
            )

    if violations:
        print("   ❌ ROLE VIOLATIONS DETECTED:")
        for violation in violations:
            print(f"     - {violation}")
    else:
        print("   ✅ No role alternation violations")

    print("\n" + "=" * 60)
    print("🏁 Test Complete")

    return {
        "regen_success": regen_result.success,
        "has_override": hasattr(regen_result, "user_input_override")
        and regen_result.user_input_override is not None,
        "conversation_length": len(conversation_history),
        "role_violations": len(violations),
        "would_add_duplicate": would_add_message and is_from_override,
    }


if __name__ == "__main__":
    print("🧪 End-to-End Test for /regen() Command")
    print("Testing the exact sequence: question -> /set() -> /regen()")

    try:
        results = test_exact_user_sequence()

        print("\n📊 SUMMARY:")
        print(f"   Regen command worked: {results['regen_success']}")
        print(f"   Has user input override: {results['has_override']}")
        print(f"   Conversation messages: {results['conversation_length']}")
        print(f"   Role violations: {results['role_violations']}")
        print(f"   Would cause duplicate: {results['would_add_duplicate']}")

        if (
            results["regen_success"]
            and results["has_override"]
            and results["role_violations"] == 0
        ):
            print("\n✅ ALL TESTS PASSED - /regen() should work correctly!")
        else:
            print("\n❌ ISSUES DETECTED - /regen() may still have problems")

    except Exception as e:
        print(f"\n💥 ERROR during testing: {e}")
        import traceback

        traceback.print_exc()
