#!/usr/bin/env python3
"""Example usage of the DatasetAnalyzer class with v1.0.0 config."""

from oumi.analyze import Analyzer
from oumi.core.configs import AnalyzerConfig, DatasetSchema, InputConfig

# First, make sure to restart your Python kernel/REPL to clear any cached imports
print("Make sure to restart your Python kernel/REPL if you're getting AttributeError")

# Example 1: Alpaca Dataset
print("=" * 60)
print("Example 1: Alpaca Dataset")
print("=" * 60)

# Create a config for Alpaca dataset
alpaca_config = AnalyzerConfig(
    analyze_version="v1.0.0",
    input=InputConfig(
        source="oumi",
        name="alpaca",
        split="train",
        schema=DatasetSchema(type="conversation"),
    ),
    verbose=True,
)

# Create an analyzer instance with config
print("Creating DatasetAnalyzer for 'alpaca' with config...")
analyzer = Analyzer(alpaca_config)

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

# Create a config for Ultrachat dataset
ultrachat_config = AnalyzerConfig(
    analyze_version="v1.0.0",
    input=InputConfig(
        source="oumi",
        name="ultrachat",
        split="train_sft",
        schema=DatasetSchema(type="conversation"),
    ),
    verbose=True,
)

# Create an analyzer instance for ultrachat with config
print("Creating DatasetAnalyzer for 'ultrachat' with config...")
ultrachat_analyzer = Analyzer(ultrachat_config)

# Print the first conversation
print("\nPrinting the first conversation:")
ultrachat_analyzer.print_conversation(0)

# Get a specific conversation
print("\nGetting conversation 3:")
ultrachat_conversation = ultrachat_analyzer.get_conversation(3)
print(f"Conversation type: {type(ultrachat_conversation)}")

# Example 3: Using analysis
print("\n" + "=" * 60)
print("Example 3: Using Analysis")
print("=" * 60)

# Analyze the alpaca dataset
results = analyzer.analyze_dataset()
print("Alpaca analysis results:")
print(f"Dataset name: {results['dataset_name']}")
print(f"Total conversations: {results['total_conversations']}")
print(f"Conversations analyzed: {results['conversations_analyzed']}")
print(
    f"Average conversation length: {results['conversation_length_stats']['mean']:.2f}"
)

# Analyze the ultrachat dataset
ultrachat_results = ultrachat_analyzer.analyze_dataset()
print("\nUltrachat analysis results:")
print(f"Dataset name: {ultrachat_results['dataset_name']}")
print(f"Total conversations: {ultrachat_results['total_conversations']}")
print(f"Conversations analyzed: {ultrachat_results['conversations_analyzed']}")
print(
    f"Average conversation length: "
    f"{ultrachat_results['conversation_length_stats']['mean']:.2f}"
)

print("\nAll examples completed successfully!")
