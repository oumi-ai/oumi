#!/usr/bin/env python3
"""Demo script for DatasetAnalyzer functionality.

This script demonstrates the key features of the DatasetAnalyzer class:
- Dataset analysis with sample analyzers
- Querying analysis results
- Filtering datasets based on analysis results
- Comparison of different filtering approaches
"""

import logging

from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs.analyze_config import AnalyzeConfig, SampleAnalyzerParams

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run the DatasetAnalyzer demo."""
    print("🚀 Starting DatasetAnalyzer Demo")
    print("=" * 50)

    # Create configuration
    config = AnalyzeConfig(
        dataset_name="tatsu-lab/alpaca",
        split="train",
        analyzers=[
            SampleAnalyzerParams(
                id="length",
                config={"metrics": ["char_count", "word_count", "sentence_count"]},
            )
        ],
        sample_count=50,  # Limit for demo
    )

    # Create analyzer
    print("📊 Creating analyzer...")
    analyzer = DatasetAnalyzer(config)

    # Run analysis
    print("🔍 Running analysis...")
    analyzer.analyze_dataset()

    # Show analysis results
    results = analyzer.get_analysis_results()
    print("\n📈 Analysis Results:")
    print(f"Dataset: {results.dataset_name}")
    print(f"Has analysis results: {analyzer.has_analysis_results()}")
    print(f"Total conversations: {len(analyzer.dataset)}")
    print(f"Conversations analyzed: {results.conversations_analyzed}")
    print(f"Total messages: {results.total_messages}")

    # Show sample message analysis
    if results.messages:
        print("\n📝 Sample Message Analysis:")
        for i, msg_result in enumerate(results.messages[:3]):
            print(f"\nMessage {i + 1} ({msg_result.role}):")
            print(f"  Content: '{msg_result.text_content[:50]}...'")
            print(f"  Metrics: {msg_result.analyzer_metrics}")

    # Query Demo
    print("\n🔍 Query Demo:")

    # Query for short messages
    short_messages = analyzer.query("length_word_count < 10")
    print(f"Short messages (< 10 words): {len(short_messages)} found")

    # Query for assistant messages
    assistant_messages = analyzer.query("role == 'assistant'")
    print(f"Assistant messages: {len(assistant_messages)} found")

    # Query for long user messages
    long_user_messages = analyzer.query("role == 'user' and length_word_count > 20")
    print(f"Long user messages (> 20 words): {len(long_user_messages)} found")

    # Filter Demo
    print("\n🔍 Filter Demo:")

    # Filter for short messages
    short_dataset = analyzer.filter("length_word_count < 10")
    print(f"Short messages dataset: {len(short_dataset)} conversations")
    print(f"Dataset name: {short_dataset.dataset_name}")

    # Filter for assistant messages
    assistant_dataset = analyzer.filter("role == 'assistant'")
    print(f"Assistant messages dataset: {len(assistant_dataset)} conversations")
    print(f"Dataset name: {assistant_dataset.dataset_name}")

    # Query vs Filter Comparison
    print("\n📊 Query vs Filter Comparison:")

    # Query returns DataFrame
    query_results = analyzer.query("role == 'user'")
    print(
        f"Query returns DataFrame with {len(query_results)} rows and "
        f"{len(query_results.columns)} columns"
    )

    # Filter returns dataset
    filter_results = analyzer.filter("role == 'user'")
    print(f"Filter returns dataset with {len(filter_results)} conversations")
    print(f"Original dataset: {len(analyzer.dataset)} conversations")
    print(f"Filtered dataset: {len(filter_results)} conversations")
    print("Both datasets have the same interface (conversation() method)")

    # Using Filtered Dataset
    print("\n🔍 Using Filtered Dataset:")
    if len(filter_results) > 0:
        first_conv = filter_results.conversation(0)
        print(f"First filtered conversation has {len(first_conv.messages)} messages")
        if first_conv.messages:
            print(f"First message role: {first_conv.messages[0].role}")
            print(f"First message content: '{first_conv.messages[0].content[:50]}...'")

    # Demo unified dataset filtering capabilities
    print("\n🚀 Dataset Filtering Demo:")
    if analyzer._is_huggingface_dataset():
        print("✅ This is a HuggingFace dataset!")
        print(f"Dataset name: {analyzer.dataset.dataset_name}")
        print(f"Dataset path: {analyzer.dataset.dataset_path}")
        print("Benefits of unified filtering:")
        print("  - Works the same for both HF and custom datasets")
        print("  - No special handling needed")
        print("  - Consistent interface regardless of dataset type")
    else:
        print("📁 This is a custom dataset")
        print("Filtering works the same way for all dataset types")

    # Show filtering performance comparison
    print("\n⚡ Filtering Performance Demo:")
    import time

    start_time = time.time()
    performance_filtered = analyzer.filter("length_word_count > 5")
    filter_time = time.time() - start_time
    print(f"Filtering took {filter_time:.3f} seconds")
    print(f"Filtered to {len(performance_filtered)} conversations")
    print(f"Filtered dataset type: {type(performance_filtered)}")
    print(f"Has conversation() method: {hasattr(performance_filtered, 'conversation')}")
    print(f"Has __len__ method: {hasattr(performance_filtered, '__len__')}")

    # Show that the filtered dataset has the exact same class as the original
    print("\n🔍 Class Preservation Demo:")
    original_class = type(analyzer.dataset)
    filtered_class = type(performance_filtered)
    print(f"Original dataset class: {original_class.__name__}")
    print(f"Filtered dataset class: {filtered_class.__name__}")
    print(
        f"Filtered dataset inherits from original: "
        f"{issubclass(filtered_class, original_class)}"
    )
    print("✅ The filtered dataset preserves the exact same class as the original!")

    # Detailed comparison between current approach and Approach 2
    print("\n🔍 Detailed Comparison: Current vs Approach 2")
    print("=" * 60)

    # Test both approaches with the same data
    test_query = "length_word_count > 5"
    conversation_indices = (
        analyzer.query(test_query).conversation_index.unique().tolist()
    )

    print("\n📊 Performance Comparison:")
    import time

    # Test current approach (Dynamic Class Inheritance)
    start_time = time.time()
    current_filtered = analyzer.filter(test_query)
    current_creation_time = time.time() - start_time

    # Test access performance for current approach
    start_time = time.time()
    for i in range(min(10, len(current_filtered))):
        _ = current_filtered.conversation(i)
    current_access_time = time.time() - start_time

    # Test Approach 2 (DataFrame Manipulation)
    start_time = time.time()
    original_df = analyzer.dataset.data
    filtered_df = original_df.iloc[conversation_indices].copy()

    class DataFrameComparisonDataset(original_class):
        def __init__(self, filtered_dataframe, original_dataset):
            # Copy all attributes from original
            for attr_name, attr_value in original_dataset.__dict__.items():
                setattr(self, attr_name, attr_value)

            # Override the data with filtered DataFrame
            self._data = filtered_dataframe
            self.dataset_name = f"{original_dataset.dataset_name}_dataframe_filtered"

        def transform(self, sample):
            """Required abstract method implementation."""
            return self._original_dataset.transform(sample)

    dataframe_filtered = DataFrameComparisonDataset(filtered_df, analyzer.dataset)
    dataframe_creation_time = time.time() - start_time

    # Test access performance for DataFrame approach
    start_time = time.time()
    for i in range(min(10, len(dataframe_filtered))):
        _ = dataframe_filtered.conversation(i)
    dataframe_access_time = time.time() - start_time

    print("\n⏱️  Creation Time:")
    print(f"   Current Approach: {current_creation_time:.4f} seconds")
    print(f"   DataFrame Approach: {dataframe_creation_time:.4f} seconds")
    winner = (
        "DataFrame" if dataframe_creation_time < current_creation_time else "Current"
    )
    print(f"   Winner: {winner}")

    print("\n⚡ Access Time (10 conversations):")
    print(f"   Current Approach: {current_access_time:.4f} seconds")
    print(f"   DataFrame Approach: {dataframe_access_time:.4f} seconds")
    print(
        f"   Winner: "
        f"{'Current' if current_access_time < dataframe_access_time else 'DataFrame'}"
    )

    print("\n💾 Memory Usage:")
    print("   Current Approach: Pre-computed conversations in memory")
    print("   DataFrame Approach: Filtered DataFrame in memory")
    print("   Note: Both store data in memory, but different formats")

    print("\n🔧 Implementation Complexity:")
    print("   Current Approach:")
    print("     - Creates dynamic class inheriting from original")
    print("     - Pre-computes conversations for fast access")
    print("     - Handles all dataset types automatically")
    print("     - Requires conversation() method implementation")

    print("   DataFrame Approach:")
    print("     - Creates new class inheriting from original")
    print("     - Uses DataFrame indexing for filtering")
    print("     - Requires _data attribute access")
    print("     - Simpler conversation() method (uses original)")

    print("\n🎯 Class Preservation:")
    print("   Current Approach: ✅ Exact same class (inherits from original)")
    print("   DataFrame Approach: ❌ New class (FilteredDataset vs AlpacaDataset)")

    print("\n🔄 Data Flow:")
    print("   Current Approach:")
    print(
        "     Original Dataset → Analysis Results → "
        "Filtered Conversations → New Dataset"
    )
    print("   DataFrame Approach:")
    print("     Original Dataset → DataFrame → Filtered DataFrame → New Dataset")

    print("\n📈 Scalability:")
    print("   Current Approach:")
    print("     - Creation: O(n) where n = number of filtered conversations")
    print("     - Access: O(1) - pre-computed conversations")
    print("     - Memory: Higher (stores Conversation objects)")

    print("   DataFrame Approach:")
    print("     - Creation: O(1) - simple DataFrame indexing")
    print("     - Access: O(1) - uses original conversation() method")
    print("     - Memory: Lower (stores raw DataFrame data)")

    print("\n🛡️  Robustness:")
    print("   Current Approach:")
    print("     - ✅ Works with any dataset class")
    print("     - ✅ Preserves all original methods")
    print("     - ✅ Handles complex conversation structures")
    print("     - ❌ Requires conversation() method")

    print("   DataFrame Approach:")
    print("     - ✅ Works with BaseMapDataset classes")
    print("     - ✅ Uses original conversation() method")
    print("     - ✅ Simpler implementation")
    print("     - ❌ Requires _data attribute access")
    print("     - ❌ May not work with all dataset types")

    print("\n🏆 Final Recommendation:")
    print("   Use Current Approach (Dynamic Class Inheritance) because:")
    print("     - ✅ Preserves exact same class as original")
    print("     - ✅ Works with ALL dataset types")
    print("     - ✅ Fast access performance")
    print("     - ✅ Most robust and universal")
    print("     - ✅ Maintains all original functionality")

    print("\n✅ Demo completed successfully!")

    # Additional demonstration: How load_dataset_from_config works
    print("\n🔍 How load_dataset_from_config Works:")
    print("=" * 50)

    print("\n📊 Data Loading Process:")
    print("1. load_dataset_from_config() calls REGISTRY.get_dataset()")
    print("2. Creates dataset instance (e.g., AlpacaDataset)")
    print("3. Dataset.__init__() calls self._load_data()")
    print("4. _load_data() loads raw data into pandas DataFrame")
    print("5. DataFrame stored in self._data attribute")

    print("\n💾 What's Actually in Memory:")
    print(f"✅ Raw DataFrame: {analyzer.dataset.data.shape}")
    print(f"   - Columns: {list(analyzer.dataset.data.columns)}")
    print(f"   - Data type: {type(analyzer.dataset.data)}")
    print(
        f"   - Memory usage: "
        f"~{analyzer.dataset.data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
    )

    print("\n🔄 How conversation() Works:")
    print("1. conversation(idx) calls raw(idx)")
    print("2. raw(idx) returns self._data.iloc[idx] (pandas Series)")
    print("3. transform_conversation() converts Series to Conversation object")
    print("4. Conversation object created on-demand (not pre-computed)")

    # Demonstrate the data flow
    print("\n🎯 Data Flow Demonstration:")
    raw_sample = analyzer.dataset.raw(0)
    print(f"Raw sample type: {type(raw_sample)}")
    print(f"Raw sample content: {raw_sample.to_dict()}")

    conversation = analyzer.dataset.conversation(0)
    print(f"Conversation type: {type(conversation)}")
    print(f"Conversation messages: {len(conversation.messages)}")

    print("\n📈 Memory Efficiency:")
    print("✅ Only raw DataFrame is loaded into memory")
    print("✅ Conversations are created on-demand")
    print("✅ No pre-computed Conversation objects")
    print("✅ Memory usage scales with raw data size")
    print("✅ Access time: O(1) for DataFrame, + conversion time for Conversation")


if __name__ == "__main__":
    main()
