#!/usr/bin/env python3
"""Example usage of the DatasetAnalyzer class."""

from oumi.analyze import DatasetAnalyzer, analyze_dataset, get_statistics

# First, make sure to restart your Python kernel/REPL to clear any cached imports
print("Make sure to restart your Python kernel/REPL if you're getting AttributeError")

# Example 1: Alpaca Dataset
print("=" * 60)
print("Example 1: Alpaca Dataset")
print("=" * 60)

# Create an analyzer instance by passing the dataset name
print("Creating DatasetAnalyzer for 'alpaca'...")
analyzer = DatasetAnalyzer("alpaca")

# Print the first conversation (this should work)
print("\nPrinting the first conversation:")
analyzer.print_conversation(0)

# Get a specific conversation
print("\nGetting conversation 5:")
conversation = analyzer.get_conversation(5)
print(f"Conversation type: {type(conversation)}")

# Example 2: Ultrachat Dataset
print("\n" + "=" * 60)
print("Example 2: Ultrachat Dataset")
print("=" * 60)

# Create an analyzer instance for ultrachat with split
print("Creating DatasetAnalyzer for 'ultrachat' with split='train'...")
ultrachat_analyzer = DatasetAnalyzer("ultrachat", split="train_sft")

# Print the first conversation
print("\nPrinting the first conversation:")
ultrachat_analyzer.print_conversation(0)

# Get a specific conversation
print("\nGetting conversation 3:")
ultrachat_conversation = ultrachat_analyzer.get_conversation(3)
print(f"Conversation type: {type(ultrachat_conversation)}")

# Example 3: Using convenience functions
print("\n" + "=" * 60)
print("Example 3: Using Convenience Functions")
print("=" * 60)

# Analyze the alpaca dataset
results = analyze_dataset("alpaca")
print(f"Alpaca analysis results: {results}")

# Get statistics for ultrachat with split
stats = get_statistics("ultrachat", split="train_sft")
print(f"Ultrachat statistics: {stats}")

print("\nAll examples completed successfully!")
