#!/usr/bin/env python3
"""Test script for vision language completions-only training."""

from oumi.builders import build_data_collator, build_tokenizer

# Build a simple model and tokenizer for testing
from oumi.core.configs import ModelParams
from oumi.core.types import ContentItem, Conversation, Message, Role
from oumi.core.types.conversation import Type

model_id = "microsoft/Phi-3-vision-128k-instruct"
model_params = ModelParams(
    model_name=model_id,
    device_map="cpu",
    trust_remote_code=True,
    chat_template="phi3-instruct",
)
tokenizer = build_tokenizer(model_params)

# Create the collator with completions-only training enabled
collator = build_data_collator(
    collator_name="vision_language_sft",
    tokenizer=tokenizer,
    processor_name=model_id,
    train_on_completions_only=True,
    response_template="<|assistant|>",  # Phi-3 uses this format
    instruction_template="<|user|>",
    trust_remote_code=True,
    max_length=512,
)

# Create a sample conversation with an image
conversation = Conversation(
    messages=[
        Message(
            role=Role.USER,
            content=[
                ContentItem(type=Type.TEXT, content="What is in this image?"),
                ContentItem(
                    type=Type.IMAGE_PATH,
                    content="/Users/oussamaelachqar/source/lema/oumi/tests/testdata/images/the_great_wave_off_kanagawa.jpg",
                ),
            ],
        ),
        Message(
            role=Role.ASSISTANT,
            content=[
                ContentItem(
                    type=Type.TEXT,
                    content="This is a PNG transparency demonstration image showing dice.",
                )
            ],
        ),
    ]
)

# Create a batch
batch = [{"conversation_json": conversation.to_json()}]

# Process the batch
try:
    result = collator(batch)
    print("✓ Collator executed successfully!")
    print(f"  Keys in result: {list(result.keys())}")
    print(f"  Input IDs shape: {result['input_ids'].shape}")
    print(f"  Labels shape: {result['labels'].shape}")

    # Check that labels are masked properly
    labels = result["labels"][0]
    input_ids = result["input_ids"][0]

    # Count non-ignored tokens in labels
    ignore_index = -100  # Standard ignore index
    non_ignored_count = (labels != ignore_index).sum().item()
    total_count = len(labels)

    print(f"  Non-ignored tokens: {non_ignored_count}/{total_count}")
    print(
        f"  Percentage masked: {((total_count - non_ignored_count) / total_count * 100):.1f}%"
    )

    # Find where non-ignored tokens start
    first_non_ignored = -1
    for i in range(len(labels)):
        if labels[i].item() != ignore_index:
            first_non_ignored = i
            break

    # Show tokens around the transition point
    if first_non_ignored > 0:
        print(
            f"\n  Tokens around assistant response (starting at index {first_non_ignored}):"
        )
        start_idx = max(0, first_non_ignored - 5)
        end_idx = min(len(input_ids), first_non_ignored + 15)

        for i in range(start_idx, end_idx):
            token_id = input_ids[i].item()
            if token_id < 0 or token_id >= len(tokenizer):
                token = f"<special:{token_id}>"
            else:
                token = tokenizer.decode([token_id])
            label = labels[i].item()
            if label != ignore_index:
                print(f"    [{i}] '{token}' -> label: {label} ✓")
            else:
                print(f"    [{i}] '{token}' -> IGNORED")

    # Show the full decoded text to understand the format
    print("\n  Full decoded text (truncated to 500 chars):")
    # Decode only positive token IDs
    valid_ids = [id.item() for id in input_ids if 0 <= id.item() < len(tokenizer)]
    decoded_text = tokenizer.decode(valid_ids)
    print(f"    {decoded_text[:500]}...")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
