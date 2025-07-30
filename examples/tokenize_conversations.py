#!/usr/bin/env python3
"""Example: Tokenizing conversations using the tokenizer.

This example demonstrates how to use the tokenizer to tokenize conversations
from the dataset in various ways.
"""

from oumi.builders import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams


def example_tokenize_conversations():
    """Example of tokenizing conversations using the tokenizer."""
    print("=== Tokenizing Conversations Example ===\n")

    # Build a tokenizer
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Create config with tokenizer
    config = AnalyzeConfig(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        tokenizer=tokenizer,
        split="train_sft",
        sample_count=3,  # Small sample for demonstration
    )

    # Create analyzer
    analyzer = DatasetAnalyzer(config)

    print(f"Dataset: {analyzer.dataset_name}")
    print(f"Tokenizer: {type(analyzer.tokenizer).__name__}")
    print(f"Sample count: {config.sample_count}")

    # Method 1: Using dataset's built-in tokenization
    print("\n--- Method 1: Dataset Built-in Tokenization ---")
    if len(analyzer.dataset) > 0 and analyzer.tokenizer is not None:
        # Get tokenized data directly from dataset
        tokenized_data = analyzer.dataset[0]
        print(f"Tokenized data keys: {list(tokenized_data.keys())}")
        print(f"Input IDs length: {len(tokenized_data['input_ids'])}")
        print(f"Attention mask length: {len(tokenized_data['attention_mask'])}")

        # Decode to see the full formatted conversation
        decoded = analyzer.tokenizer.decode(tokenized_data["input_ids"])
        print("Decoded conversation (first 300 chars):")
        print(f"  {decoded[:300]}...")

    # Method 2: Manual tokenization of conversations
    print("\n--- Method 2: Manual Tokenization ---")
    if len(analyzer.dataset) > 0 and analyzer.tokenizer is not None:
        # Get a conversation
        conversation = analyzer.dataset.conversation(0)
        print(f"Conversation has {len(conversation.messages)} messages")

        # Tokenize the conversation manually
        tokenized_manual = analyzer.dataset.tokenize(conversation)
        print(f"Manual tokenization keys: {list(tokenized_manual.keys())}")
        print(f"Manual input IDs length: {len(tokenized_manual['input_ids'])}")

        # Compare with dataset tokenization
        if "input_ids" in tokenized_data and "input_ids" in tokenized_manual:
            print(
                f"Tokenization matches: "
                f"{tokenized_data['input_ids'] == tokenized_manual['input_ids']}"
            )

    # Method 3: Tokenize individual messages
    print("\n--- Method 3: Tokenizing Individual Messages ---")
    if len(analyzer.dataset) > 0 and analyzer.tokenizer is not None:
        conversation = analyzer.dataset.conversation(0)

        for i, message in enumerate(conversation.messages):
            print(f"\nMessage {i + 1} ({message.role.value}):")
            print(f"  Content: {message.content[:100]}...")

            # Tokenize just this message
            message_tokens = analyzer.tokenizer.encode(
                message.content, add_special_tokens=False
            )
            print(f"  Token count: {len(message_tokens)}")
            print(f"  First 10 tokens: {message_tokens[:10]}")

            # Decode back to text
            decoded_message = analyzer.tokenizer.decode(message_tokens)
            print(f"  Decoded: {decoded_message[:100]}...")

    # Method 4: Tokenize with different parameters
    print("\n--- Method 4: Tokenization with Different Parameters ---")
    if len(analyzer.dataset) > 0 and analyzer.tokenizer is not None:
        conversation = analyzer.dataset.conversation(0)

        # Tokenize with special tokens
        with_special = analyzer.tokenizer.encode(
            conversation.messages[0].content, add_special_tokens=True
        )
        print(f"With special tokens: {len(with_special)} tokens")

        # Tokenize without special tokens
        without_special = analyzer.tokenizer.encode(
            conversation.messages[0].content, add_special_tokens=False
        )
        print(f"Without special tokens: {len(without_special)} tokens")

        # Tokenize with truncation
        truncated = analyzer.tokenizer.encode(
            conversation.messages[0].content,
            add_special_tokens=True,
            max_length=50,
            truncation=True,
        )
        print(f"Truncated to 50 tokens: {len(truncated)} tokens")

    # Method 5: Batch tokenization
    print("\n--- Method 5: Batch Tokenization ---")
    if len(analyzer.dataset) > 1 and analyzer.tokenizer is not None:
        # Get multiple conversations
        conversations = [
            analyzer.dataset.conversation(i)
            for i in range(min(3, len(analyzer.dataset)))
        ]

        # Extract text content from first message of each conversation
        texts = [conv.messages[0].content for conv in conversations]

        # Batch tokenize
        batch_tokens = analyzer.tokenizer(
            texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=100,
            return_tensors="pt",
        )

        print(f"Batch tokenized {len(texts)} conversations")
        print(f"Batch shape: {batch_tokens['input_ids'].shape}")
        print(f"Attention mask shape: {batch_tokens['attention_mask'].shape}")

        # Decode batch
        for i, tokens in enumerate(batch_tokens["input_ids"]):
            decoded_batch = analyzer.tokenizer.decode(tokens)
            print(f"  Conversation {i + 1}: {decoded_batch[:100]}...")

    return analyzer


def example_tokenize_with_different_models():
    """Example of tokenizing with different model tokenizers."""
    print("\n=== Tokenizing with Different Models ===\n")

    # Test with different tokenizers
    models = [
        ("openai-community/gpt2", {"pad_token": "<|endoftext|>"}),
        ("microsoft/Phi-3-mini-4k-instruct", {}),
    ]

    for model_name, tokenizer_kwargs in models:
        print(f"\n--- Testing with {model_name} ---")

        try:
            # Build tokenizer
            model_params = ModelParams(
                model_name=model_name, tokenizer_kwargs=tokenizer_kwargs
            )
            tokenizer = build_tokenizer(model_params)

            # Create config
            config = AnalyzeConfig(
                dataset_name="HuggingFaceH4/ultrachat_200k",
                tokenizer=tokenizer,
                split="train_sft",
                sample_count=1,
            )

            # Create analyzer
            analyzer = DatasetAnalyzer(config)

            if len(analyzer.dataset) > 0 and analyzer.tokenizer is not None:
                tokenized = analyzer.dataset[0]

                print(f"  Tokenizer: {type(tokenizer).__name__}")
                print(f"  Input IDs length: {len(tokenized['input_ids'])}")
                chat_template_preview = (
                    tokenizer.chat_template[:50] if tokenizer.chat_template else "None"
                )
                print(f"  Chat template: {chat_template_preview}...")

                # Show a snippet of the tokenized conversation
                decoded = tokenizer.decode(tokenized["input_ids"][:100])
                print(f"  Sample: {decoded[:150]}...")

        except Exception as e:
            print(f"  Error with {model_name}: {e}")


if __name__ == "__main__":
    # Run examples
    analyzer = example_tokenize_conversations()
    example_tokenize_with_different_models()

    print("\n" + "=" * 60)
    print("Tokenization examples completed successfully!")
    print("=" * 60)
