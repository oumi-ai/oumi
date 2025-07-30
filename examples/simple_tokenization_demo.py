#!/usr/bin/env python3
"""Simple demo: How to tokenize conversations manually.

This example shows the basic steps to tokenize conversations using the tokenizer.
"""

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams


def simple_tokenization_demo():
    """Simple demo of tokenizing conversations."""
    print("=== Simple Tokenization Demo ===\n")

    # 1. Build a tokenizer
    print("1. Building tokenizer...")
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)
    print(f"   Tokenizer: {type(tokenizer).__name__}")

    # 2. Create analyzer with tokenizer
    print("\n2. Creating dataset analyzer...")
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=1,
    )
    analyzer = DatasetAnalyzer(config)

    # 3. Get a conversation
    print("\n3. Getting a conversation...")
    conversation = analyzer.dataset.conversation(0)
    print(f"   Conversation has {len(conversation.messages)} messages")

    # 4. Show the conversation structure
    print("\n4. Conversation structure:")
    for i, message in enumerate(conversation.messages):
        print(f"   Message {i + 1} ({message.role.value}): {message.content[:50]}...")

    # 5. Tokenize the entire conversation
    print("\n5. Tokenizing entire conversation...")
    tokenized = analyzer.dataset.tokenize(conversation)
    print(f"   Input IDs length: {len(tokenized['input_ids'])}")
    print(f"   Attention mask length: {len(tokenized['attention_mask'])}")

    # 6. Decode to see the formatted conversation
    print("\n6. Decoded conversation (first 200 chars):")
    decoded = tokenizer.decode(tokenized["input_ids"])
    print(f"   {decoded[:200]}...")

    # 7. Tokenize individual messages
    print("\n7. Tokenizing individual messages:")
    for i, message in enumerate(conversation.messages[:3]):  # Show first 3 messages
        tokens = tokenizer.encode(message.content, add_special_tokens=False)
        print(f"   Message {i + 1} ({message.role.value}): {len(tokens)} tokens")
        print(f"   First 10 tokens: {tokens[:10]}")

    # 8. Compare with dataset's built-in tokenization
    print("\n8. Comparing with dataset's built-in tokenization...")
    dataset_tokenized = analyzer.dataset[0]
    print(f"   Dataset tokenization length: {len(dataset_tokenized['input_ids'])}")
    print(f"   Manual tokenization length: {len(tokenized['input_ids'])}")
    print(f"   Match: {dataset_tokenized['input_ids'] == tokenized['input_ids']}")

    return analyzer, conversation, tokenized


def tokenize_custom_conversation():
    """Example of tokenizing a custom conversation."""
    print("\n=== Custom Conversation Tokenization ===\n")

    # Build tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create a custom conversation
    from oumi.core.types.conversation import Conversation, Message, Role

    custom_conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="What is the capital of France?"),
            Message(role=Role.ASSISTANT, content="The capital of France is Paris."),
            Message(role=Role.USER, content="What is the population of Paris?"),
            Message(
                role=Role.ASSISTANT,
                content=(
                    "As of 2021, the population of Paris is approximately "
                    "2.2 million people."
                ),
            ),
        ]
    )

    print("Custom conversation:")
    for i, message in enumerate(custom_conversation.messages):
        print(f"  {message.role.value}: {message.content}")

    # Tokenize using the tokenizer directly
    print("\nTokenizing custom conversation...")

    # Method 1: Using the tokenizer's apply_chat_template
    if tokenizer.chat_template:
        # Convert conversation to the format expected by apply_chat_template
        messages_dict = [
            {"role": msg.role.value, "content": msg.content}
            for msg in custom_conversation.messages
        ]

        formatted = tokenizer.apply_chat_template(
            messages_dict, tokenize=True, return_dict=True
        )
        print("Method 1 - Chat template tokenization:")

        # Handle the return type safely
        if isinstance(formatted, dict) and "input_ids" in formatted:
            input_ids = formatted["input_ids"]
            print(f"  Input IDs length: {len(input_ids)}")
            print(f"  Decoded: {tokenizer.decode(input_ids)[:100]}...")
        else:
            print(f"  Unexpected format: {type(formatted)}")

    # Method 2: Manual tokenization
    print("\nMethod 2 - Manual tokenization:")
    for i, message in enumerate(custom_conversation.messages):
        # Ensure content is a string
        content = (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )
        tokens = tokenizer.encode(content, add_special_tokens=False)
        print(f"  Message {i + 1} ({message.role.value}): {len(tokens)} tokens")

    return custom_conversation


if __name__ == "__main__":
    # Run the demos
    analyzer, conversation, tokenized = simple_tokenization_demo()
    custom_conv = tokenize_custom_conversation()

    print("\n" + "=" * 60)
    print("Tokenization demo completed!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    print(f"- Dataset conversation: {len(conversation.messages)} messages")
    print(f"- Tokenized length: {len(tokenized['input_ids'])} tokens")
    print(f"- Custom conversation: {len(custom_conv.messages)} messages")
    print(f"- Tokenizer: {type(analyzer.tokenizer).__name__}")
