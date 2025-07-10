#!/usr/bin/env python3
"""Simple test script to verify DatasetAnalyzer functionality."""

from oumi.analyze import Analyzer
from oumi.core.configs import (
    AggregationMetrics,
    AnalyzerConfig,
    DatasetSchema,
    InputConfig,
    LanguageDetectionConfig,
    OutputConfig,
    SampleLevelMetrics,
)
from oumi.core.configs.analyzer_config import (
    ConversationAggregationMetricsConfig,
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
                analysis_output="ultrachat_analysis",
                aggregation_output="ultrachat_aggregations",
                conversation_level_output="ultrachat_conversation_level",
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
            aggregation_metrics=AggregationMetrics(),
            conversation_aggregation_metrics=ConversationAggregationMetricsConfig(
                enabled=True,
                turn_count=True,
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
            aggregation_results = results["aggregation_results"]
            basic_stats = aggregation_results.get("basic_stats", {})
            print(
                f"Total conversations in dataset: "
                f"{basic_stats.get('total_conversations', 0)}"
            )
            print(
                f"Conversations analyzed: "
                f"{basic_stats.get('conversations_analyzed', 0)}"
            )
            print(f"Total messages: {basic_stats.get('total_messages', 0)}")
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


def test_aggregation_flags():
    """Test aggregation metrics flags (basic_stats, conversation_stats)."""
    try:
        print("=" * 60)
        print("Testing Aggregation Metrics Flags")
        print("=" * 60)

        # Test with basic_stats disabled
        print("Testing with basic_stats disabled...")
        config_basic_disabled = AnalyzerConfig(
            input=InputConfig(
                name="tatsu-lab/alpaca",
                split="train",
                max_conversations=10,  # Small subset for testing
                schema=DatasetSchema(type="conversation"),
            ),
            outputs=OutputConfig(
                path="./test_results",
                analysis_output="test_basic_disabled_sample",
                aggregation_output="test_basic_disabled_agg",
                conversation_level_output="test_basic_disabled_conv",
                save_format="json",
            ),
            aggregation_metrics=AggregationMetrics(
                basic_stats=False,  # Disable basic stats
                conversation_stats=True,
            ),
            conversation_aggregation_metrics=ConversationAggregationMetricsConfig(
                enabled=True,
                turn_count=True,
            ),
            verbose=True,
        )

        analyzer_basic_disabled = Analyzer(config_basic_disabled)
        results_basic_disabled = analyzer_basic_disabled.analyze_dataset()

        # Check that basic_stats is not in aggregation results
        aggregation_results = results_basic_disabled.get("aggregation_results", {})
        if "basic_stats" not in aggregation_results:
            print("✅ basic_stats correctly excluded when disabled")
        else:
            print("❌ basic_stats should not be present when disabled")
            print(f"Found: {aggregation_results.keys()}")

        # Check that conversation_stats is still present
        if "conversation_stats" in aggregation_results:
            print("✅ conversation_stats correctly included when enabled")
        else:
            print("❌ conversation_stats should be present when enabled")

        # Test with conversation_stats disabled
        print("\nTesting with conversation_stats disabled...")
        config_conv_disabled = AnalyzerConfig(
            input=InputConfig(
                name="tatsu-lab/alpaca",
                split="train",
                max_conversations=10,
                schema=DatasetSchema(type="conversation"),
            ),
            outputs=OutputConfig(
                path="./test_results",
                analysis_output="test_conv_disabled_sample",
                aggregation_output="test_conv_disabled_agg",
                conversation_level_output="test_conv_disabled_conv",
                save_format="json",
            ),
            aggregation_metrics=AggregationMetrics(
                basic_stats=True,
                conversation_stats=False,  # Disable conversation stats
            ),
            conversation_aggregation_metrics=ConversationAggregationMetricsConfig(
                enabled=True,
                turn_count=True,
            ),
            verbose=True,
        )

        analyzer_conv_disabled = Analyzer(config_conv_disabled)
        results_conv_disabled = analyzer_conv_disabled.analyze_dataset()

        # Check that basic_stats is present
        aggregation_results = results_conv_disabled.get("aggregation_results", {})
        if "basic_stats" in aggregation_results:
            print("✅ basic_stats correctly included when enabled")
        else:
            print("❌ basic_stats should be present when enabled")

        # Check that conversation_stats is not present
        if "conversation_stats" not in aggregation_results:
            print("✅ conversation_stats correctly excluded when disabled")
        else:
            print("❌ conversation_stats should not be present when disabled")
            print(f"Found: {aggregation_results.keys()}")

        # Test with both disabled
        print("\nTesting with both basic_stats and conversation_stats disabled...")
        config_both_disabled = AnalyzerConfig(
            input=InputConfig(
                name="tatsu-lab/alpaca",
                split="train",
                max_conversations=10,
                schema=DatasetSchema(type="conversation"),
            ),
            outputs=OutputConfig(
                path="./test_results",
                analysis_output="test_both_disabled_sample",
                aggregation_output="test_both_disabled_agg",
                conversation_level_output="test_both_disabled_conv",
                save_format="json",
            ),
            aggregation_metrics=AggregationMetrics(
                basic_stats=False,
                conversation_stats=False,
            ),
            conversation_aggregation_metrics=ConversationAggregationMetricsConfig(
                enabled=True,
                turn_count=True,
            ),
            verbose=True,
        )

        analyzer_both_disabled = Analyzer(config_both_disabled)
        results_both_disabled = analyzer_both_disabled.analyze_dataset()

        # Check that aggregation results is empty
        aggregation_results = results_both_disabled.get("aggregation_results", {})
        if not aggregation_results:
            print("✅ aggregation_results correctly empty when both flags disabled")
        else:
            print("❌ aggregation_results should be empty when both flags disabled")
            print(f"Found: {aggregation_results.keys()}")

        print("\n" + "=" * 60)
        print("All aggregation flags tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during aggregation flags test: {e}")
        import traceback

        traceback.print_exc()


def test_yaml_config_loading():
    """Test loading and using the YAML config file directly."""
    try:
        print("=" * 60)
        print("Testing YAML Config File Loading")
        print("=" * 60)

        # Load the YAML config file
        from oumi.core.configs import AnalyzerConfig

        yaml_config_path = "configs/analyzer/analyzer_config_v1_example.yaml"
        print(f"Loading YAML config from: {yaml_config_path}")

        # Use the from_yaml method from BaseConfig
        config = AnalyzerConfig.from_yaml(yaml_config_path)

        # Override dataset and limit for testing
        config.input.name = "tatsu-lab/alpaca"
        config.input.max_conversations = 5
        config.input.split = "train"  # Fix split for alpaca dataset
        config.outputs.path = "./test_results"

        print("Config loaded successfully:")
        print(f"  - Analysis output prefix: {config.outputs.analysis_output}")
        print(f"  - Aggregation output prefix: {config.outputs.aggregation_output}")

        # Create analyzer and run analysis
        analyzer = Analyzer(config)
        analyzer.analyze_dataset()

        # Check that files were created with correct prefixes
        from pathlib import Path

        expected_prefixes = [
            config.outputs.analysis_output + "_sample_level",
            config.outputs.aggregation_output,
            config.outputs.conversation_level_output,
        ]

        print("\nChecking generated files:")
        output_dir = Path(config.outputs.path)
        for prefix in expected_prefixes:
            pattern = f"{prefix}_*.{config.outputs.save_format}"
            matching_files = list(output_dir.glob(pattern))
            if matching_files:
                print(f"✅ Found files for prefix '{prefix}':")
                for file in matching_files:
                    print(f"    - {file.name}")
            else:
                print(f"❌ No files found for prefix '{prefix}'")

        print("\n" + "=" * 60)
        print("YAML config loading test completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during YAML config test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_analyzer()
    test_v1_config()
    test_config_validation()
    test_aggregation_flags()
    test_yaml_config_loading()
