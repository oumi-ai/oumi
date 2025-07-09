#!/usr/bin/env python3
"""Simple test script to verify DatasetAnalyzer functionality."""

from oumi.analyze import Analyzer
from oumi.core.configs import (
    AggregationMetrics,
    AnalyzerConfig,
    DatasetSchema,
    InputConfig,
    LanguageAggregationConfig,
    LanguageDetectionConfig,
    OutputConfig,
    SampleLevelMetrics,
)
from oumi.core.configs.analyzer_config import (
    LengthMetricsConfig,
    SafetyMetricsConfig,
    SafetyTypeConfig,
)


def test_analyzer():
    """Test the DatasetAnalyzer class with config-based initialization."""
    try:
        # Test Alpaca dataset
        print("=" * 60)
        print("Testing Alpaca Dataset (Config Mode)")
        print("=" * 60)

        # Create an analyzer instance with config
        print("Creating DatasetAnalyzer for 'tatsu-lab/alpaca' with config...")
        config = AnalyzerConfig(
            input=InputConfig(
                name="tatsu-lab/alpaca",
                split="train",
                schema=DatasetSchema(type="conversation"),
            ),
            verbose=True,
        )
        analyzer = Analyzer(config)

        # Test getting dataset size
        print(f"\nDataset size: {analyzer.get_dataset_size()} conversations")

        # Test getting a conversation
        print("\nGetting conversation 0...")
        conversation = analyzer.get_conversation(0)
        print(f"Conversation type: {type(conversation)}")

        # Test getting conversation length
        print(f"Conversation 0 length: {analyzer.get_conversation_length(0)} messages")

        # Test printing a conversation
        print("\nPrinting conversation 0...")
        analyzer.print_conversation(0)

        # Test getting another conversation
        print("\nGetting conversation 5...")
        conversation5 = analyzer.get_conversation(5)
        print(f"Conversation 5 type: {type(conversation5)}")
        print(f"Conversation 5 length: {analyzer.get_conversation_length(5)} messages")

        print("\nAlpaca test completed successfully!")

        # Test Ultrachat dataset
        print("\n" + "=" * 60)
        print("Testing Ultrachat Dataset (Config Mode)")
        print("=" * 60)

        # Create an analyzer instance for ultrachat with config
        print(
            "Creating DatasetAnalyzer for 'huggingfaceh4/ultrachat_200k' with config..."
        )
        ultrachat_config = AnalyzerConfig(
            input=InputConfig(
                name="huggingfaceh4/ultrachat_200k",
                split="train_sft",
                schema=DatasetSchema(type="conversation"),
            ),
            verbose=True,
        )
        ultrachat_analyzer = Analyzer(ultrachat_config)

        # Test getting dataset size
        print(f"\nDataset size: {ultrachat_analyzer.get_dataset_size()} conversations")

        # Test getting a conversation
        print("\nGetting conversation 0...")
        ultrachat_conversation = ultrachat_analyzer.get_conversation(0)
        print(f"Conversation type: {type(ultrachat_conversation)}")

        # Test getting conversation length
        print(
            "Conversation 0 length: "
            + str(ultrachat_analyzer.get_conversation_length(0))
            + " messages"
        )

        # Test printing a conversation
        print("\nPrinting conversation 0...")
        ultrachat_analyzer.print_conversation(0)

        # Test getting another conversation
        print("\nGetting conversation 3...")
        ultrachat_conversation3 = ultrachat_analyzer.get_conversation(3)
        print(f"Conversation 3 type: {type(ultrachat_conversation3)}")
        print(
            "Conversation 3 length: "
            + str(ultrachat_analyzer.get_conversation_length(3))
            + " messages"
        )

        print("\nUltrachat test completed successfully!")

        print("\n" + "=" * 60)
        print("All config mode tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback

        traceback.print_exc()


def test_v1_config():
    """Test the v1.0.0 configuration structure."""
    try:
        print("=" * 60)
        print("Testing v1.0.0 Configuration")
        print("=" * 60)

        # Test basic v1.0.0 config with Alpaca dataset
        print("Creating basic v1.0.0 config for 'tatsu-lab/alpaca'...")
        config = AnalyzerConfig(
            input=InputConfig(
                name="tatsu-lab/alpaca",
                split="train",
                schema=DatasetSchema(type="conversation"),
            ),
            verbose=True,
        )

        analyzer = Analyzer(config)

        # Test getting dataset size
        print(f"\nDataset size: {analyzer.get_dataset_size()} conversations")

        # Test getting a conversation
        print("\nGetting conversation 0...")
        conversation = analyzer.get_conversation(0)
        print(f"Conversation type: {type(conversation)}")

        # Test getting conversation length
        print(f"Conversation 0 length: {analyzer.get_conversation_length(0)} messages")

        # Test printing a conversation
        print("\nPrinting conversation 0...")
        analyzer.print_conversation(0)

        print("\nBasic v1.0.0 config test completed successfully!")

        # Test comprehensive v1.0.0 config with Ultrachat dataset
        print("\n" + "=" * 60)
        print("Testing Comprehensive v1.0.0 Configuration")
        print("=" * 60)

        comprehensive_config = AnalyzerConfig(
            input=InputConfig(
                name="huggingfaceh4/ultrachat_200k",
                split="train_sft",
                max_conversations=100,  # Limit to first 100 conversations for testing
                schema=DatasetSchema(type="conversation"),
            ),
            outputs=OutputConfig(
                path="./test_results",
                analysis_output="ultrachat_analysis.json",
                aggregation_output="ultrachat_aggregations.json",
                save_format="json",
            ),
            sample_level_metrics=SampleLevelMetrics(
                language=LanguageDetectionConfig(
                    enabled=True,
                    confidence_threshold=0.2,
                    top_k=3,
                    multilingual_flag={"enabled": True, "min_num_languages": 2},
                ),
                length=LengthMetricsConfig(
                    enabled=True,
                    char_count=True,
                    word_count=True,
                    sentence_count=True,
                    token_count=False,
                ),
                safety=SafetyMetricsConfig(
                    enabled=True,
                    profanity=SafetyTypeConfig(
                        enabled=True,
                        include_default=True,
                        custom_keywords=[],
                        custom_regexes=[],
                    ),
                    slurs=SafetyTypeConfig(
                        enabled=True,
                        include_default=True,
                        custom_keywords=[],
                        custom_regexes=[],
                    ),
                    explicit=SafetyTypeConfig(
                        enabled=True,
                        include_default=True,
                        custom_keywords=[],
                        custom_regexes=[],
                    ),
                    hate_speech=SafetyTypeConfig(
                        enabled=True,
                        include_default=True,
                        custom_keywords=[],
                        custom_regexes=[],
                    ),
                    pii=SafetyTypeConfig(
                        enabled=True,
                        include_default=True,
                        custom_keywords=[],
                        custom_regexes=[],
                    ),
                ),
            ),
            aggregation_metrics=AggregationMetrics(
                language=LanguageAggregationConfig(
                    distribution={
                        "enabled": True,
                        "min_samples": 10,
                        "report_top_n": 10,
                        "include_other_bucket": True,
                    },
                    minority_alert={"enabled": True, "threshold_percent": 5.0},
                    confidence_statistics={
                        "enabled": True,
                        "stats": ["mean", "stddev", "percentile_10", "percentile_90"],
                    },
                    multilingual_samples={
                        "enabled": True,
                        "common_language_pairs": True,
                    },
                )
            ),
            verbose=True,
        )

        comprehensive_analyzer = Analyzer(comprehensive_config)

        # Test getting dataset size
        print(
            f"\nDataset size: {comprehensive_analyzer.get_dataset_size()} conversations"
        )

        # Test getting a conversation
        print("\nGetting conversation 0...")
        ultrachat_conversation = comprehensive_analyzer.get_conversation(0)
        print(f"Conversation type: {type(ultrachat_conversation)}")

        # Test getting conversation length
        print(
            f"Conversation 0 length: "
            f"{comprehensive_analyzer.get_conversation_length(0)} messages"
        )

        # Test printing a conversation
        print("\nPrinting conversation 0...")
        comprehensive_analyzer.print_conversation(0)

        # Test analysis
        print("\nRunning analysis...")
        results = comprehensive_analyzer.analyze_dataset()
        print(f"Analysis completed: {results.get('status', 'success')}")
        print(f"Dataset name: {results.get('dataset_name', 'unknown')}")

        # Handle new two-step analysis structure
        if "sample_level_results" in results and "aggregation_results" in results:
            sample_results = results["sample_level_results"]
            aggregation_results = results["aggregation_results"]
            print(
                f"Total conversations in dataset: "
                f"{sample_results.get('total_conversations', 0)}"
            )
            print(
                f"Conversations analyzed: "
                f"{sample_results.get('conversations_analyzed', 0)}"
            )
            print(f"Total messages: {sample_results.get('total_messages', 0)}")
            role_analysis = aggregation_results.get("role_analysis", {})
            avg_messages = role_analysis.get("avg_messages_per_conversation", 0)
            print(f"Average messages per conversation: {avg_messages}")
        else:
            # Fallback for old structure
            print(f"Total conversations: {results.get('total_conversations', 0)}")

        print("\nComprehensive v1.0.0 config test completed successfully!")

        print("\n" + "=" * 60)
        print("All v1.0.0 tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during v1.0.0 test: {e}")
        import traceback

        traceback.print_exc()


def test_config_validation():
    """Test configuration validation."""
    try:
        print("=" * 60)
        print("Testing Configuration Validation")
        print("=" * 60)

        # Test missing name
        print("Testing missing name...")
        try:
            config_without_name = AnalyzerConfig(input=InputConfig(name=""))
            # The error should be caught when we try to use the config
            _ = Analyzer(config_without_name)
            print("❌ Should have failed but didn't")
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")

        # Test invalid confidence threshold
        print("\nTesting invalid confidence threshold...")
        try:
            _ = AnalyzerConfig(
                input=InputConfig(name="test"),
                sample_level_metrics=SampleLevelMetrics(
                    language=LanguageDetectionConfig(confidence_threshold=1.5)
                ),
            )
            print("❌ Should have failed but didn't")
        except ValueError as e:
            print(f"✅ Correctly caught error: {e}")

        # Test unregistered dataset
        print("\nTesting unregistered dataset...")
        try:
            config = AnalyzerConfig(
                input=InputConfig(name="unregistered-dataset"),
            )
            _ = Analyzer(config)
            print("❌ Should have failed but didn't")
        except NotImplementedError as e:
            print(f"✅ Correctly caught error: {e}")

        print("\n" + "=" * 60)
        print("All validation tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during validation test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_analyzer()
    test_v1_config()
    test_config_validation()
