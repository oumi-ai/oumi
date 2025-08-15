"""Integration tests for analyzing multiple different dataset types."""

import pytest

from oumi.builders.models import build_tokenizer
from oumi.core.analyze import DatasetAnalyzer
from oumi.core.configs import AnalyzeConfig, ModelParams, SampleAnalyzerParams


@pytest.mark.e2e  # Mark as e2e since it downloads and analyzes real datasets
def test_analyze_multiple_datasets():
    """Test that dataset analyzer can read and analyze different dataset types.

    This test verifies that the analyze functionality can successfully:
    1. Load different types of datasets from HuggingFace Hub
    2. Process the first 30 samples from each dataset
    3. Generate analysis results that can be queried
    """

    # Create length analyzer configuration similar to notebook
    length_analyzer = SampleAnalyzerParams(
        id="length",
        params={
            "char_count": True,
            "word_count": True,
            "sentence_count": True,
            "token_count": True,
        },
    )

    # Create model params and tokenizer similar to notebook
    model_params = ModelParams(
        model_name="openai-community/gpt2",
        tokenizer_kwargs={"pad_token": "<|endoftext|>"},
    )
    tokenizer = build_tokenizer(model_params)

    # Define the datasets to test - these are the datasets mentioned in the user query
    # Some datasets may not have "train" split, so we specify the split explicitly
    datasets_to_test = [
        ("oumi-ai/walton-multimodal-cold-start-r1-format", "train"),
        ("oumi-ai/s1-vis-mid-resize", "train"),
        ("oumi-ai/limo-vis-mid-resize", "train"),
        ("oumi-ai/multimodal-open-r1-8192-filtered-mid-ic", "train"),
        ("openmed-community/synthetic-neurology-conversations", "train"),
        ("oumi-ai/MM-MathInstruct-to-r1-format-filtered", "train"),
        ("Squad", "train"),
    ]

    for dataset_name, split in datasets_to_test:
        config = AnalyzeConfig(
            dataset_name=dataset_name,
            split=split,
            sample_count=30,  # Analyze first 30 samples
            tokenizer=tokenizer,
            output_path="./test_results",
            analyzers=[length_analyzer],
            processor_name="Salesforce/blip2-opt-2.7b",
            processor_kwargs={},
            trust_remote_code=True,
        )

        try:
            # Initialize analyzer
            analyzer = DatasetAnalyzer(config)

            # Run analysis
            analyzer.analyze_dataset()

            # Verify results were generated
            results = analyzer.analysis_results
            assert results is not None, f"No analysis results for {dataset_name}"
            assert results.dataset_name == dataset_name, (
                f"Dataset name mismatch for {dataset_name}"
            )
            assert results.conversations_analyzed >= 30, (
                f"Too few conversations analyzed for {dataset_name}: "
                f"expected = 30, got {results.conversations_analyzed}"
            )

            # Verify we can query the results - use a simple query to get all results
            analysis_df = analyzer.analysis_df
            assert analysis_df is not None, f"No results dataframe for {dataset_name}"
            assert len(analysis_df) > 0, f"No results dataframe for {dataset_name}"

            # Verify basic structure of results
            assert "conversation_id" in analysis_df.columns, (
                f"Missing conversation_id column for {dataset_name}"
            )

            print(
                f"✓ Successfully analyzed {dataset_name} ({split} split): "
                f"{results.conversations_analyzed} conversations processed"
            )

        except Exception as e:
            pytest.fail(f"Failed to analyze dataset {dataset_name}: {e}")
