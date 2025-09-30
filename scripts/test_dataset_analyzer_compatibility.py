#!/usr/bin/env python3
"""Manual test script for DatasetAnalyzer compatibility with all datasets.

This script allows developers to test dataset analyzer compatibility
without running the full pytest suite. Useful for:

1. Quick validation during development
2. Testing specific dataset types
3. Debugging schema detection issues
4. Generating compatibility reports

For automated testing and CI/CD, use the integration tests:
    pytest tests/integration/datasets/test_dataset_analyzer_all_datasets.py

Usage:
    python scripts/test_dataset_analyzer_compatibility.py --help
    python scripts/test_dataset_analyzer_compatibility.py --category sft
    python scripts/test_dataset_analyzer_compatibility.py --dataset alpaca
    python scripts/test_dataset_analyzer_compatibility.py --all --sample-count 1
    python scripts/test_dataset_analyzer_compatibility.py --quick --category pretraining
    python scripts/test_dataset_analyzer_compatibility.py --all --exclude-gated
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer

from oumi.core.analyze.dataset_analyzer import DatasetAnalyzer
from oumi.core.configs.analyze_config import AnalyzeConfig, DatasetSource
from oumi.core.datasets import (
    BasePretrainingDataset,
    BaseSftDataset,
    VisionLanguageSftDataset,
)
from oumi.core.registry import REGISTRY, RegistryType

# List of datasets that require HuggingFace access or are known to be gated
GATED_DATASETS = {
    "bigcode/starcoderdata",
    "bigcode/the-stack",
    "nampdn-ai/tiny-textbooks",
    # Add more as discovered
}


def categorize_dataset_class(dataset_class) -> str:
    """Categorize dataset by its base class."""
    if issubclass(dataset_class, BaseSftDataset):
        return "sft"
    elif issubclass(dataset_class, VisionLanguageSftDataset):
        return "vision_sft"
    elif issubclass(dataset_class, BasePretrainingDataset):
        return "pretraining"
    elif "Dpo" in dataset_class.__name__ or "DPO" in dataset_class.__name__:
        return "dpo"
    elif "Kto" in dataset_class.__name__ or "KTO" in dataset_class.__name__:
        return "kto"
    elif "Grpo" in dataset_class.__name__ or "GRPO" in dataset_class.__name__:
        return "grpo"
    else:
        return "other"


def _get_dataset_config(dataset_key: str) -> dict:
    """Get dataset-specific configuration including split, subset, and other params.

    Note: Dataset size limiting is now handled automatically by the DatasetAnalyzer
    based on the sample_count parameter in AnalyzeConfig.
    """
    # Dataset-specific configurations (same as in integration test)
    dataset_configs = {
        # Evaluation datasets that only have eval/test splits
        "tatsu-lab/alpaca_eval": {"split": "eval"},
        "nvidia/ChatRAG-Bench": {"split": "test"},
        # Vision datasets with specific splits and processor requirements
        "merve/vqav2-small": {
            "split": "validation",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "HuggingFaceM4/Docmatix": {"split": "test", "subset": "zero-shot-exp"},
        "hiyouga/geometry3k": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "huggingfaceh4/llava-instruct-mix-vsft": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "lmms-lab/multimodal-open-r1-8k-verified": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "mnist_sft": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "allenai/pixmo-ask-model-anything": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "allenai/pixmo-cap": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        "allenai/pixmo-cap-qa": {
            "split": "train",
            "processor_name": "microsoft/Phi-3-mini-4k-instruct",
        },
        # Datasets with specific splits
        "nlphuji/flickr30k": {"split": "test"},  # Only has test split
        "huggingfaceh4/ultrachat_200k": {
            "split": "train_sft"
        },  # Needs train_sft not train
        # GRPO datasets - use train split explicitly
        "oumi-ai/berrybench-v0.1.1": {"split": "train"},
        "d1shs0ap/countdown": {"split": "train"},
        "oumi-ai/oumi-letter-count": {"split": "train"},
        "oumi-ai/oumi-letter-count-clean": {"split": "train"},
        "trl-lib/tldr": {"split": "train"},
        # KTO datasets
        "trl-lib/kto-mix-14k": {"split": "train"},
        # Datasets that need specific configs/subsets
        "openai/gsm8k": {"split": "train", "subset": "main"},
        "allenai/c4": {"split": "train", "subset": "en"},
        "allenai/dolma": {
            "split": "train",
            "subset": "v1_7",
            "trust_remote_code": True,
        },
        # Datasets requiring trust_remote_code
        "eleutherai/pile": {"split": "train", "trust_remote_code": True},
        "togethercomputer/redpajama-data-1t": {
            "split": "train",
            "subset": "common_crawl",
            "trust_remote_code": True,
        },
        "togethercomputer/redpajama-data-v2": {
            "split": "train",
            "subset": "default",
            "trust_remote_code": True,
        },
        "coco_captions": {"split": "train", "trust_remote_code": True},
        # Datasets with specific subsets
        "wikimedia/wikipedia": {"split": "train", "subset": "20231101.en"},
        "salesforce/wikitext": {"split": "train", "subset": "wikitext-2-raw-v1"},
        "huggingfacem4/the_cauldron": {"split": "train", "subset": "ai2d"},
        "huggingfacem4/docmatix": {"split": "train", "subset": "zero-shot-exp"},
        # Debug datasets
        "debug_sft": {"split": "train"},
        "debug_pretraining": {"split": "train"},
        "debug_dpo": {"split": "train"},
        "debug_kto": {"split": "train"},
    }

    # Return specific config if mapped, otherwise default to train split
    return dataset_configs.get(dataset_key, {"split": "train"})


def get_all_datasets() -> dict[str, tuple[type, str]]:
    """Get all registered datasets with their categories."""
    datasets = {}
    for key, dataset_class in REGISTRY.get_all(RegistryType.DATASET).items():
        category = categorize_dataset_class(dataset_class)
        datasets[key] = (dataset_class, category)
    return datasets


def test_single_dataset(
    dataset_key: str,
    dataset_class: type,
    category: str,
    sample_count: int = 1,
    test_conversion: bool = False,
    test_full_analysis: bool = False,
    quick_mode: bool = False,
) -> dict:
    """Test a single dataset with the analyzer."""
    result = {
        "dataset_key": dataset_key,
        "category": category,
        "success": False,
        "detected_type": None,
        "schema_keys": [],
        "error": None,
        "message_df_shape": None,
    }

    try:
        # Quick mode: only test schema detection without creating datasets
        if quick_mode:
            # Use the dataset class directly for type detection
            dataset_class_bases = [base.__name__ for base in dataset_class.__mro__]

            # Detect dataset type based on class hierarchy
            if any(
                base in dataset_class_bases
                for base in [
                    "BaseSftDataset",
                    "VisionLanguageSftDataset",
                    "BaseExperimentalGrpoDataset",
                ]
            ):
                detected_type = "oumi"  # All convert to conversation format
            elif "BasePretrainingDataset" in dataset_class_bases:
                detected_type = "pretraining"
            elif any(
                base in dataset_class_bases
                for base in ["BaseDpoDataset", "VisionLanguageDpoDataset"]
            ):
                detected_type = "dpo"
            elif "BaseKtoDataset" in dataset_class_bases:
                detected_type = "kto"
            else:
                detected_type = "oumi"  # Default fallback

            # Get schema for the detected type
            from oumi.utils.analysis_utils import get_schema_for_format

            schema = get_schema_for_format(detected_type)

            result["detected_type"] = detected_type
            result["schema_keys"] = list(schema.keys())
            result["success"] = True
            return result

        # Get dataset-specific config first
        dataset_config = _get_dataset_config(dataset_key)

        # Apply dataset slicing for map datasets to limit download size
        if sample_count and sample_count > 0:
            # Check if this is a map dataset (not iterable)
            from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

            is_iterable_dataset = issubclass(dataset_class, BaseIterableDataset)

            if not is_iterable_dataset:
                # For map datasets, modify split to include slicing
                if "split" in dataset_config and "[" not in dataset_config["split"]:
                    slice_size = sample_count * 10  # Use larger slice for safety
                    dataset_config["split"] = (
                        f"{dataset_config['split']}[:{slice_size}]"
                    )
                    print(
                        f"  📊 Limiting dataset to {slice_size} items: "
                        f"{dataset_config['split']}"
                    )

        # Create dataset with appropriate config
        if category == "pretraining":
            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
            dataset = dataset_class(
                tokenizer=tokenizer,
                seq_length=64,
                stream=True,  # Enable streaming for large datasets
                **dataset_config,
            )
        elif category in ["sft", "vision_sft", "grpo"]:
            try:
                dataset = dataset_class(**dataset_config)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/Phi-3-mini-4k-instruct"
                )
                dataset = dataset_class(tokenizer=tokenizer, **dataset_config)
        else:
            dataset = dataset_class(**dataset_config)

        # Test analyzer
        config = AnalyzeConfig(
            dataset_source=DatasetSource.DIRECT,
            sample_count=sample_count,
            analyzers=[],
        )

        analyzer = DatasetAnalyzer(config=config, dataset=dataset)
        result["detected_type"] = analyzer._detect_dataset_type()
        schema = analyzer._get_schema_for_dataset()
        result["schema_keys"] = list(schema.keys())

        if test_conversion:
            analyzer.analyze_dataset()
            if analyzer._message_df is not None:
                result["message_df_shape"] = analyzer._message_df.shape

        if test_full_analysis:
            # Import here to avoid circular imports
            from oumi.core.configs.analyze_config import SampleAnalyzerParams

            # Create length analyzer for full analysis testing
            length_analyzer = SampleAnalyzerParams(
                id="length",
                params={
                    "char_count": True,
                    "word_count": True,
                    "sentence_count": True,
                    "token_count": True,
                },
            )

            tokenizer_config = {
                "model_name": "openai-community/gpt2",
                "tokenizer_kwargs": {"pad_token": "<|endoftext|>"},
            }

            full_config = AnalyzeConfig(
                dataset_name=dataset_key,
                dataset_source=DatasetSource.CONFIG,
                sample_count=sample_count,
                tokenizer_config=tokenizer_config,
                output_path=f"./test_results_{dataset_key.replace('/', '_')}",
                analyzers=[length_analyzer],
                **dataset_config,
            )

            # Run full analysis
            full_analyzer = DatasetAnalyzer(full_config)
            full_analyzer.analyze_dataset()

            # Validate results
            if full_analyzer._analysis_results:
                try:
                    analysis_summary = full_analyzer.analysis_summary
                    if (
                        "message_level_summary" in analysis_summary
                        and "length" in analysis_summary["message_level_summary"]
                    ):
                        result["analysis_metrics"] = list(
                            analysis_summary["message_level_summary"]["length"].keys()
                        )
                    else:
                        result["analysis_metrics"] = [
                            "pipeline_worked_but_no_length_metrics"
                        ]
                except Exception:
                    result["analysis_metrics"] = ["pipeline_worked_but_summary_failed"]
            else:
                result["analysis_metrics"] = []

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def print_results_table(results: list[dict]):
    """Print results in a formatted table."""
    print(
        f"\n{'Dataset':<25} {'Category':<12} {'Status':<8} "
        f"{'Detected':<12} {'Schema Keys'}"
    )
    print("-" * 80)

    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        detected = result["detected_type"] or "N/A"
        schema_keys = ", ".join(result["schema_keys"][:3])  # Show first 3 keys
        if len(result["schema_keys"]) > 3:
            schema_keys += "..."

        print(
            f"{result['dataset_key']:<25} {result['category']:<12} {status:<8} "
            f"{detected:<12} {schema_keys}"
        )

        if not result["success"] and result["error"]:
            print(f"  Error: {result['error']}")


def main():
    """Main function to run dataset analyzer compatibility tests."""
    parser = argparse.ArgumentParser(description="Test DatasetAnalyzer compatibility")
    parser.add_argument("--all", action="store_true", help="Test all datasets")
    parser.add_argument(
        "--category", help="Test specific category (sft, dpo, pretraining, etc.)"
    )
    parser.add_argument("--dataset", help="Test specific dataset by key")
    parser.add_argument(
        "--sample-count", type=int, default=1, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--test-conversion", action="store_true", help="Test full data conversion"
    )
    parser.add_argument(
        "--test-full-analysis",
        action="store_true",
        help="Test complete analysis pipeline with analyzers",
    )
    parser.add_argument(
        "--exclude-debug", action="store_true", help="Exclude debug datasets"
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="List all available datasets"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only test schema detection (no data conversion or analysis)",
    )
    parser.add_argument(
        "--exclude-gated",
        action="store_true",
        help="Exclude gated datasets that require HuggingFace access",
    )

    args = parser.parse_args()

    all_datasets = get_all_datasets()

    if args.list_datasets:
        print("Available datasets by category:")
        by_category = {}
        for key, (_, category) in all_datasets.items():
            by_category.setdefault(category, []).append(key)

        for category, datasets in sorted(by_category.items()):
            print(f"\n{category.upper()} ({len(datasets)} datasets):")
            for dataset in sorted(datasets):
                print(f"  - {dataset}")
        return

    # Filter datasets based on arguments
    datasets_to_test = {}

    if args.dataset:
        if args.dataset in all_datasets:
            datasets_to_test[args.dataset] = all_datasets[args.dataset]
        else:
            print(f"Dataset '{args.dataset}' not found!")
            return
    elif args.category:
        datasets_to_test = {
            key: (cls, cat)
            for key, (cls, cat) in all_datasets.items()
            if cat == args.category
        }
    elif args.all:
        datasets_to_test = all_datasets
    else:
        # Default: test debug datasets for quick validation
        datasets_to_test = {
            key: (cls, cat)
            for key, (cls, cat) in all_datasets.items()
            if "debug" in key.lower()
        }

    if args.exclude_debug:
        datasets_to_test = {
            key: (cls, cat)
            for key, (cls, cat) in datasets_to_test.items()
            if "debug" not in key.lower()
        }

    if args.exclude_gated:
        datasets_to_test = {
            key: (cls, cat)
            for key, (cls, cat) in datasets_to_test.items()
            if key not in GATED_DATASETS
        }

    if not datasets_to_test:
        print("No datasets to test!")
        return

    print(f"Testing {len(datasets_to_test)} datasets...")
    if args.quick:
        print(
            "🚀 Quick mode enabled - only testing schema detection (no dataset loading)"
        )
    elif args.test_conversion:
        print("⚠️  Full conversion testing enabled - this may take a long time!")
    elif args.test_full_analysis:
        print(
            "⚠️  Full analysis pipeline testing enabled - "
            "this will take a very long time!"
        )

    results = []
    for i, (dataset_key, (dataset_class, category)) in enumerate(
        datasets_to_test.items(), 1
    ):
        print(f"[{i}/{len(datasets_to_test)}] Testing {dataset_key} ({category})...")

        result = test_single_dataset(
            dataset_key,
            dataset_class,
            category,
            args.sample_count,
            args.test_conversion,
            args.test_full_analysis,
            args.quick,
        )
        results.append(result)

        if result["success"]:
            print(f"  ✅ Success: {result['detected_type']} schema")
        else:
            print(f"  ❌ Failed: {result['error']}")

    # Print summary
    print_results_table(results)

    success_count = sum(1 for r in results if r["success"])
    print(f"\nSummary: {success_count}/{len(results)} datasets passed")

    if success_count < len(results):
        print("\nFailed datasets:")
        for result in results:
            if not result["success"]:
                print(f"  - {result['dataset_key']}: {result['error']}")


if __name__ == "__main__":
    main()
