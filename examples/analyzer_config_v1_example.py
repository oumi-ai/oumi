#!/usr/bin/env python3
"""Example usage of the AnalyzerConfig v1.0.0 with comprehensive analysis features."""

from oumi.analyze import Analyzer
from oumi.core.configs import (
    AggregationMetrics,
    AnalyzerConfig,
    DatasetSchema,
    InputConfig,
    LanguageAggregationConfig,
    LanguageDetectionConfig,
    OutputConfig,
    PreprocessingConfig,
    SampleLevelMetrics,
)


def example_basic_v1_config():
    """Example 1: Basic v1.0.0 configuration."""
    print("=" * 60)
    print("Example 1: Basic v1.0.0 configuration")
    print("=" * 60)

    # Create a basic v1.0.0 config
    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(
            source="oumi",
            name="alpaca",
            split="train",
            schema=DatasetSchema(
                type="conversation",
                fields={
                    "text_field": "text",
                    "conversation_field": "messages",
                    "conversation_id_field": "id",
                    "role_field": "role",
                    "content_field": "content",
                },
            ),
        ),
        preprocessing=PreprocessingConfig(
            normalize_whitespace=True, lowercase=False, remove_special_chars=False
        ),
        outputs=OutputConfig(
            analysis_output="alpaca_analysis.parquet",
            aggregation_output="alpaca_aggregations.json",
            save_format="json",
        ),
        sample_level_metrics=SampleLevelMetrics(
            language=LanguageDetectionConfig(
                enabled=True,
                confidence_threshold=0.2,
                top_k=3,
                multilingual_flag={"enabled": True, "min_num_languages": 2},
            )
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
                multilingual_samples={"enabled": True, "common_language_pairs": True},
            )
        ),
        verbose=True,
    )

    print(f"Created v1.0.0 config for dataset: {config.input.name}")
    print(f"Source: {config.input.source}")
    print(f"Schema type: {config.input.schema.type}")
    print(f"Language detection enabled: {config.sample_level_metrics.language.enabled}")
    print(f"Output format: {config.outputs.save_format}")

    return config


def example_huggingface_source():
    """Example 2: HuggingFace source configuration."""
    print("\n" + "=" * 60)
    print("Example 2: HuggingFace source configuration")
    print("=" * 60)

    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(
            source="huggingface",
            name="tatsu-lab/alpaca",
            split="train",
            schema=DatasetSchema(type="conversation"),
        ),
        preprocessing=PreprocessingConfig(
            normalize_whitespace=True,
            lowercase=True,  # Convert to lowercase for analysis
            remove_special_chars=True,  # Remove special characters
        ),
        outputs=OutputConfig(
            analysis_output="hf_alpaca_analysis.parquet",
            aggregation_output="hf_alpaca_aggregations.json",
        ),
        sample_level_metrics=SampleLevelMetrics(
            language=LanguageDetectionConfig(
                enabled=True,
                confidence_threshold=0.5,  # Higher confidence threshold
                top_k=5,
            )
        ),
        verbose=True,
    )

    print(f"Created HuggingFace config for: {config.input.name}")
    print(
        f"Preprocessing: lowercase={config.preprocessing.lowercase}, "
        f"remove_special_chars={config.preprocessing.remove_special_chars}"
    )
    print(
        f"Language confidence threshold: "
        f"{config.sample_level_metrics.language.confidence_threshold}"
    )

    return config


def example_custom_source():
    """Example 3: Custom source configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom source configuration")
    print("=" * 60)

    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(
            source="custom",
            path="/path/to/your/dataset.jsonl",
            format="jsonl",
            split="train",
            schema=DatasetSchema(
                type="single_turn",
                fields={
                    "text_field": "text",
                    "conversation_field": "messages",
                    "conversation_id_field": "id",
                },
            ),
        ),
        preprocessing=PreprocessingConfig(
            normalize_whitespace=True, lowercase=False, remove_special_chars=False
        ),
        outputs=OutputConfig(
            analysis_output="custom_analysis.parquet",
            aggregation_output="custom_aggregations.json",
        ),
        sample_level_metrics=SampleLevelMetrics(
            language=LanguageDetectionConfig(
                enabled=True, confidence_threshold=0.3, top_k=3
            )
        ),
        verbose=True,
    )

    print(f"Created custom source config for: {config.input.path}")
    print(f"File format: {config.input.format}")
    print(f"Schema type: {config.input.schema.type}")

    return config


def example_from_yaml():
    """Example 4: Loading configuration from YAML."""
    print("\n" + "=" * 60)
    print("Example 4: Loading configuration from YAML")
    print("=" * 60)

    # Example YAML content
    yaml_content = """
analyze_version: "v1.0.0"

input:
  source: "oumi"
  name: "ultrachat"
  split: "train_sft"
  schema:
    type: "conversation"
    fields:
      text_field: "text"
      conversation_field: "messages"
      conversation_id_field: "id"
      role_field: "role"
      content_field: "content"

preprocessing:
  normalize_whitespace: true
  lowercase: false
  remove_special_chars: false

outputs:
  analysis_output: "ultrachat_analysis.parquet"
  aggregation_output: "ultrachat_aggregations.json"
  save_format: "json"

sample_level_metrics:
  language:
    enabled: true
    confidence_threshold: 0.2
    top_k: 3
    multilingual_flag:
      enabled: true
      min_num_languages: 2

aggregation_metrics:
  language:
    distribution:
      enabled: true
      min_samples: 10
      report_top_n: 10
      include_other_bucket: true
    minority_alert:
      enabled: true
      threshold_percent: 5.0
    confidence_statistics:
      enabled: true
      stats: [mean, stddev, percentile_10, percentile_90]
    multilingual_samples:
      enabled: true
      common_language_pairs: true

verbose: true
"""

    # Load config from YAML
    config = AnalyzerConfig.from_str(yaml_content)

    print("Loaded config from YAML:")
    print(f"Version: {config.analyze_version}")
    print(f"Dataset: {config.input.name}")
    print(f"Source: {config.input.source}")
    print(f"Language detection: {config.sample_level_metrics.language.enabled}")
    print(
        f"Multilingual flag: "
        f"{config.sample_level_metrics.language.multilingual_flag['enabled']}"
    )

    return config


def example_validation():
    """Example 5: Configuration validation."""
    print("\n" + "=" * 60)
    print("Example 5: Configuration validation")
    print("=" * 60)

    # Test invalid configurations
    invalid_configs = [
        {
            "name": "Invalid source",
            "config": {"input": {"source": "invalid_source", "name": "test"}},
        },
        {
            "name": "Missing name for oumi source",
            "config": {"input": {"source": "oumi"}},
        },
        {
            "name": "Missing path for custom source",
            "config": {"input": {"source": "custom"}},
        },
        {
            "name": "Invalid schema type",
            "config": {
                "input": {
                    "source": "oumi",
                    "name": "test",
                    "schema": {"type": "invalid"},
                }
            },
        },
        {
            "name": "Invalid confidence threshold",
            "config": {
                "input": {"source": "oumi", "name": "test"},
                "sample_level_metrics": {"language": {"confidence_threshold": 1.5}},
            },
        },
    ]

    for test in invalid_configs:
        try:
            _ = AnalyzerConfig(**test["config"])
            print(f"❌ {test['name']}: Should have failed but didn't")
        except ValueError as e:
            print(f"✅ {test['name']}: Correctly caught error - {e}")
        except Exception as e:
            print(f"❌ {test['name']}: Unexpected error - {e}")


def example_usage_with_analyzer():
    """Example 6: Using the config with the Analyzer class."""
    print("\n" + "=" * 60)
    print("Example 6: Using the config with the Analyzer class")
    print("=" * 60)

    # Create a simple config for testing
    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(source="oumi", name="alpaca", split="train"),
        verbose=True,
    )

    print(f"Created config for: {config.input.name}")
    print(f"Version: {config.analyze_version}")

    try:
        # Create analyzer with config
        analyzer = Analyzer(config)
        print("✅ Successfully created analyzer with v1.0.0 config")
        print(f"Dataset name: {analyzer.dataset_name}")
        print(f"Dataset size: {analyzer.get_dataset_size()}")

        # Perform analysis
        results = analyzer.analyze_dataset()
        print(f"Analysis completed: {results.get('status', 'success')}")

    except Exception as e:
        print(f"❌ Error using config with analyzer: {e}")


if __name__ == "__main__":
    print("AnalyzerConfig v1.0.0 Examples")
    print("=" * 60)

    try:
        example_basic_v1_config()
        example_huggingface_source()
        example_custom_source()
        example_from_yaml()
        example_validation()
        example_usage_with_analyzer()

        print("\n" + "=" * 60)
        print("All v1.0.0 examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()
