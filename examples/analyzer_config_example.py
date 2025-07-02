#!/usr/bin/env python3
"""Example usage of the AnalyzerConfig with the updated analyze.py functionality."""

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


def example_basic_usage():
    """Example 1: Basic usage with AnalyzerConfig."""
    print("=" * 60)
    print("Example 1: Basic usage with AnalyzerConfig")
    print("=" * 60)

    # Create an AnalyzerConfig manually
    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(
            source="oumi",
            name="alpaca",
            split="train",
            schema=DatasetSchema(type="conversation"),
        ),
        verbose=True,
    )

    # Create analyzer with config
    analyzer = Analyzer(config)

    # Perform analysis
    results = analyzer.analyze_dataset()

    print(f"Analysis results: {results}")
    print(f"Dataset name: {results['dataset_name']}")
    print(f"Total conversations: {results['total_conversations']}")
    print(f"Conversations analyzed: {results['conversations_analyzed']}")
    print(
        f"Average conversation length: "
        f"{results['conversation_length_stats']['mean']:.2f}"
    )


def example_comprehensive_config():
    """Example 2: Using comprehensive configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Using comprehensive configuration")
    print("=" * 60)

    # Create a comprehensive config
    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(
            source="oumi",
            name="ultrachat",
            split="train_sft",
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
            analysis_output="ultrachat_analysis.parquet",
            aggregation_output="ultrachat_aggregations.json",
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

    print(f"Created comprehensive config for dataset: {config.input.name}")
    print(f"Source: {config.input.source}")
    print(f"Split: {config.input.split}")
    print(f"Schema type: {config.input.schema.type}")

    # Use the config
    analyzer = Analyzer(config)
    results = analyzer.analyze_dataset()

    print(f"Analysis completed for {results['conversations_analyzed']} conversations")


def example_yaml_config():
    """Example 3: Loading config from YAML file."""
    print("\n" + "=" * 60)
    print("Example 3: Loading config from YAML file")
    print("=" * 60)

    # Create a YAML config string
    yaml_config = """
analyze_version: "v1.0.0"
input:
  source: "oumi"
  name: "alpaca"
  split: "train"
  schema:
    type: "conversation"
    fields: {}

preprocessing:
  normalize_whitespace: true
  lowercase: false
  remove_special_chars: false

outputs:
  analysis_output: "alpaca_analysis.parquet"
  aggregation_output: "alpaca_aggregations.json"
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
      stats: ["mean", "stddev", "percentile_10", "percentile_90"]
    multilingual_samples:
      enabled: true
      common_language_pairs: true

verbose: true
"""

    # Load config from YAML string
    config = AnalyzerConfig.from_str(yaml_config)

    print("Loaded config from YAML:")
    print(f"Dataset: {config.input.name}")
    print(f"Source: {config.input.source}")
    print(f"Schema type: {config.input.schema.type}")

    # Use the config
    analyzer = Analyzer(config)
    results = analyzer.analyze_dataset()

    print(f"Analysis completed for {results['conversations_analyzed']} conversations")


def example_huggingface_dataset():
    """Example 4: Using HuggingFace dataset."""
    print("\n" + "=" * 60)
    print("Example 4: Using HuggingFace dataset")
    print("=" * 60)

    # Create config for HuggingFace dataset
    config = AnalyzerConfig(
        analyze_version="v1.0.0",
        input=InputConfig(
            source="huggingface",
            name="tatsu-lab/alpaca",
            split="train",
            schema=DatasetSchema(type="conversation"),
        ),
        verbose=True,
    )

    print(f"Created config for HuggingFace dataset: {config.input.name}")

    # Use the config
    analyzer = Analyzer(config)
    results = analyzer.analyze_dataset()

    print(f"Analysis completed for {results['conversations_analyzed']} conversations")


if __name__ == "__main__":
    print("AnalyzerConfig Examples")
    print("=" * 60)

    try:
        example_basic_usage()
        example_comprehensive_config()
        example_yaml_config()
        example_huggingface_dataset()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()
