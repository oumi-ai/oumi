# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import jinja2
import numpy as np
import pandas as pd
from tqdm import tqdm

from oumi.builders.models import build_chat_template
from oumi.core.configs.analyze_config import AnalyzeConfig
from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset
from oumi.core.datasets.base_map_dataset import BaseMapDataset
from oumi.core.registry.registry import REGISTRY
from oumi.core.types.conversation import Conversation

logger = logging.getLogger(__name__)


def load_dataset_from_config(
    config: AnalyzeConfig, tokenizer: Any | None = None
) -> BaseMapDataset | BaseIterableDataset:
    """Load dataset based on configuration.

    This function loads datasets directly from the registry for analysis purposes.
    If a tokenizer is provided, it will be passed to the dataset constructor.

    For custom datasets, it supports loading from local files using
    TextSftJsonLinesDataset for text data and VLJsonlinesDataset for
    vision-language data.

    Args:
        config: Configuration object containing dataset parameters
        tokenizer: Optional tokenizer to use with the dataset

    Returns:
        Loaded dataset
    """
    dataset_name = config.dataset_name
    split = config.split
    subset = config.subset
    dataset_path = config.dataset_path

    if not dataset_name and not dataset_path:
        raise ValueError("Either dataset_name or dataset_path must be provided")

    # Handle custom dataset loading from local files
    if dataset_path:
        return _load_custom_dataset_from_path(dataset_path, tokenizer, config)

    # Handle registered dataset loading
    try:
        # Load dataset from the REGISTRY
        if dataset_name is None:
            raise ValueError("dataset_name cannot be None for registered datasets")
        dataset_class = REGISTRY.get_dataset(dataset_name, subset=subset)

        if dataset_class is not None:
            # Check if this is an iterable dataset that supports streaming
            import inspect

            from oumi.core.datasets.base_iterable_dataset import BaseIterableDataset

            # Ensure dataset_class is actually a class before using issubclass
            is_iterable_dataset = inspect.isclass(dataset_class) and issubclass(
                dataset_class, BaseIterableDataset
            )

            # For iterable datasets, force streaming mode to avoid downloading all
            if is_iterable_dataset:
                logger.info(
                    f"Using streaming mode for iterable dataset: {dataset_name}"
                )
                # Don't modify split for iterable datasets - streaming handles limiting
            else:
                # For map datasets, modify split to include slicing if sample_count set
                if config.sample_count is not None and config.sample_count > 0:
                    # Use a larger slice (10x sample_count) to ensure enough data
                    # after any filtering that might happen in the dataset
                    slice_size = config.sample_count * 10
                    if "[" not in split:  # Only add slicing if not already present
                        split = f"{split}[:{slice_size}]"

            # Prepare dataset constructor arguments
            dataset_kwargs = {
                "dataset_name": dataset_name,
                "dataset_path": None,
                "split": split,
                "subset": subset,
                "trust_remote_code": config.trust_remote_code,
            }

            # Force streaming for iterable datasets
            if is_iterable_dataset:
                dataset_kwargs["stream"] = True

            # Add tokenizer if provided
            if tokenizer is not None:
                dataset_kwargs["tokenizer"] = tokenizer

            # Add processor parameters for vision-language datasets
            if config.processor_name:
                dataset_kwargs["processor_name"] = config.processor_name
                dataset_kwargs["processor_kwargs"] = config.processor_kwargs
                dataset_kwargs["trust_remote_code"] = config.trust_remote_code

            # Add required parameters for pretraining datasets
            if is_iterable_dataset:
                # Import here to avoid circular imports
                from oumi.core.datasets.base_pretraining_dataset import (
                    BasePretrainingDataset,
                )

                if inspect.isclass(dataset_class) and issubclass(
                    dataset_class, BasePretrainingDataset
                ):
                    # Pretraining datasets require seq_length and tokenizer
                    if "seq_length" not in dataset_kwargs:
                        dataset_kwargs["seq_length"] = 64  # Default sequence length
                    if tokenizer is None:
                        # Create a default tokenizer if none provided
                        from oumi.builders.models import build_tokenizer
                        from oumi.core.configs.params.model_params import ModelParams

                        model_params = ModelParams(
                            model_name="openai-community/gpt2",
                            tokenizer_kwargs={"pad_token": "<|endoftext|>"},
                        )
                        dataset_kwargs["tokenizer"] = build_tokenizer(model_params)

            # Load registered dataset with parameters
            # Check if dataset_class is callable (class or mock for testing)
            if not (inspect.isclass(dataset_class) or callable(dataset_class)):
                raise TypeError(
                    f"Expected class or callable, got {type(dataset_class)} for "
                    f"dataset {dataset_name}"
                )
            dataset = dataset_class(**dataset_kwargs)

            # Ensure we return a supported dataset type
            if isinstance(dataset, BaseMapDataset | BaseIterableDataset):
                return dataset
            else:
                raise NotImplementedError(
                    f"Dataset type {type(dataset)} is not supported for analysis. "
                    "Please use a dataset that inherits from BaseMapDataset or "
                    "BaseIterableDataset."
                )
        else:
            # TODO: Implement HuggingFace Hub loading
            raise NotImplementedError(
                f"Dataset '{dataset_name}' is not registered in the REGISTRY. "
                "Loading from HuggingFace Hub is not yet implemented."
            )

    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def _load_custom_dataset_from_path(
    dataset_path: str,
    tokenizer: Any | None,
    config: AnalyzeConfig,
) -> BaseMapDataset:
    """Load a custom dataset from a local file path.

    Args:
        dataset_path: Path to the dataset file
        tokenizer: Optional tokenizer to use with the dataset
        config: Configuration object containing additional parameters

    Returns:
        Loaded dataset (TextSftJsonLinesDataset or VLJsonlinesDataset)
    """
    # Import here to avoid circular imports
    from oumi.datasets.sft.sft_jsonlines import TextSftJsonLinesDataset
    from oumi.datasets.vision_language.vision_jsonlines import VLJsonlinesDataset

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if not path.is_file():
        raise ValueError(
            f"Dataset path must be a file, not a directory: {dataset_path}"
        )

    is_multimodal = config.processor_name is not None

    if is_multimodal:
        dataset_kwargs = {
            "dataset_path": str(path),
            "tokenizer": tokenizer,
            "processor_name": config.processor_name,
            "processor_kwargs": config.processor_kwargs,
            "trust_remote_code": config.trust_remote_code,
        }
        dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if v is not None}
        dataset = VLJsonlinesDataset(**dataset_kwargs)
        logger.info(f"Loaded vision-language dataset from: {dataset_path}")
        return dataset
    else:
        dataset_kwargs_text: dict[str, Any] = {
            "dataset_path": str(path),
        }
        if tokenizer is not None:
            dataset_kwargs_text["tokenizer"] = tokenizer
        dataset = TextSftJsonLinesDataset(**dataset_kwargs_text)
        logger.info(f"Loaded text dataset from: {dataset_path}")
        return dataset


def compute_statistics(series: pd.Series, decimal_precision: int = 2) -> dict[str, Any]:
    """Compute statistics for a pandas Series.

    This utility function handles edge cases like empty series or single-element
    series, ensuring that standard deviation is 0.0 for single values instead
    of NaN.

    Args:
        series: Pandas Series containing numeric values
        decimal_precision: Number of decimal places for rounding

    Returns:
        Dictionary with computed statistics (count, mean, std, min, max, median, sum)
    """
    if series.empty:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "median": 0.0,
            "sum": 0.0,
        }

    # Convert boolean series to int for statistics (True=1, False=0)
    if pd.api.types.is_bool_dtype(series):
        series = series.astype(int)

    if len(series) == 1:
        single_value = round(float(series.iloc[0]), decimal_precision)
        return {
            "count": 1,
            "mean": single_value,
            "std": 0.0,  # Standard deviation is 0 for single value
            "min": single_value,
            "max": single_value,
            "median": single_value,
            "sum": single_value,
        }

    return {
        "count": len(series),
        "mean": round(float(series.mean()), decimal_precision),
        "std": round(float(series.std()), decimal_precision),
        "min": round(float(series.min()), decimal_precision),
        "max": round(float(series.max()), decimal_precision),
        "median": round(float(series.median()), decimal_precision),
        "sum": round(float(series.sum()), decimal_precision),
    }


def render_conversation_as_text(
    conversation: Conversation,
    separator: str = "\n",
    template_name: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render a full conversation as a single text string.

    This extracts only text content from messages and formats them either using
    a simple "ROLE: content" format or by applying a chat template.

    Args:
        conversation: The conversation to render
        separator: String to separate messages (default: newline).
            Only used when template_name is None.
        template_name: Optional name of a chat template to use (e.g., 'default',
            'llama3-instruct', 'chat_ml').
            If None, uses simple "ROLE: content" format.
        add_generation_prompt: Whether to append generation prompt when using
            a template. Only used when template_name is provided. Default: False.

    Returns:
        Full conversation rendered as text

    Note:
        This is different from Conversation.__repr__() which includes message IDs
        and uses repr() for content items (showing <IMAGE_BINARY> for images).
        This function extracts only text content and is optimized for analysis.

        When using a chat template, the conversation is rendered exactly as the model
        would see it during training/inference.
    """
    if template_name is not None:
        # Use chat template rendering
        template_str = build_chat_template(template_name)
        template = jinja2.Template(template_str)

        # Convert conversation to format expected by chat templates
        messages = []
        for message in conversation.messages:
            text = message.compute_flattened_text_content()
            messages.append({"role": message.role.value, "content": text})

        # Render using the template
        return template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )
    else:
        # Use simple format (original implementation)
        message_texts = []
        for message in conversation.messages:
            # Get text content from message using existing method
            text = message.compute_flattened_text_content()

            # Format as "ROLE: content"
            role_str = message.role.value.upper()
            message_texts.append(f"{role_str}: {text}")

        return separator.join(message_texts)


# =============================================================================
# Multi-Modal Distribution Analysis
# =============================================================================


class DistributionType(str, Enum):
    """Type of distribution detected in data."""

    UNIMODAL = "unimodal"
    BIMODAL = "bimodal"
    MULTIMODAL = "multimodal"
    UNIFORM = "uniform"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class ModeStatistics:
    """Statistics for a single mode in a distribution."""

    mode_id: int
    mean: float
    std: float
    count: int
    weight: float  # Proportion of samples (0-1)


@dataclass
class DistributionAnalysisResult:
    """Result of distribution analysis including mode detection."""

    distribution_type: DistributionType
    num_modes: int
    global_statistics: dict
    mode_statistics: list[ModeStatistics] = field(default_factory=list)
    mode_assignments: list[int] | None = None  # Per-sample mode assignment
    confidence: float = 0.0  # Mode separation confidence (0-1)


def detect_distribution_type(
    series: pd.Series,
    max_components: int = 5,
    min_samples: int = 50,
    cv_threshold: float = 0.5,
) -> DistributionAnalysisResult:
    """Detect if a distribution is unimodal, bimodal, or multimodal using GMM.

    Uses Gaussian Mixture Models with BIC (Bayesian Information Criterion) to
    automatically determine the optimal number of components.

    Args:
        series: Pandas Series containing numeric values
        max_components: Maximum number of components to test
        min_samples: Minimum samples required for GMM analysis
        cv_threshold: CV below this threshold skips GMM (assumes unimodal)

    Returns:
        DistributionAnalysisResult with detected distribution type and statistics
    """
    # Handle empty or small series
    clean_series = series.dropna()
    if len(clean_series) < 2:
        return DistributionAnalysisResult(
            distribution_type=DistributionType.INSUFFICIENT_DATA,
            num_modes=0,
            global_statistics=compute_statistics(series),
        )

    global_stats = compute_statistics(clean_series)

    # Not enough data for GMM - use simple statistics
    if len(clean_series) < min_samples:
        return DistributionAnalysisResult(
            distribution_type=DistributionType.UNIMODAL,
            num_modes=1,
            global_statistics=global_stats,
            mode_statistics=[
                ModeStatistics(
                    mode_id=0,
                    mean=global_stats["mean"],
                    std=global_stats["std"],
                    count=global_stats["count"],
                    weight=1.0,
                )
            ],
            confidence=1.0,
        )

    # Pre-screening: if CV is low, distribution is likely unimodal
    mean = global_stats["mean"]
    std = global_stats["std"]
    cv = std / abs(mean) if mean != 0 else 0

    if cv < cv_threshold:
        return DistributionAnalysisResult(
            distribution_type=DistributionType.UNIMODAL,
            num_modes=1,
            global_statistics=global_stats,
            mode_statistics=[
                ModeStatistics(
                    mode_id=0,
                    mean=mean,
                    std=std,
                    count=len(clean_series),
                    weight=1.0,
                )
            ],
            confidence=1.0,
        )

    # Run GMM with BIC to determine optimal number of components
    try:
        from sklearn.mixture import GaussianMixture

        X = clean_series.to_numpy().reshape(-1, 1)

        best_n_components = 1
        best_bic = float("inf")
        best_gmm = None

        # Test 1 to max_components, select model with lowest BIC
        for n_components in range(1, min(max_components + 1, len(clean_series) // 10)):
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=42,
                n_init=3,
                max_iter=200,
            )
            gmm.fit(X)
            bic = gmm.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_n_components = n_components
                best_gmm = gmm

        if best_gmm is None:
            # Fallback to unimodal if GMM fitting failed
            return DistributionAnalysisResult(
                distribution_type=DistributionType.UNIMODAL,
                num_modes=1,
                global_statistics=global_stats,
                mode_statistics=[
                    ModeStatistics(
                        mode_id=0,
                        mean=mean,
                        std=std,
                        count=len(clean_series),
                        weight=1.0,
                    )
                ],
                confidence=1.0,
            )

        # Get mode assignments and compute per-mode statistics
        labels = best_gmm.predict(X)
        mode_stats = compute_mode_statistics(
            clean_series,
            labels,
            best_gmm.weights_,  # type: ignore
        )

        # Determine distribution type
        if best_n_components == 1:
            dist_type = DistributionType.UNIMODAL
        elif best_n_components == 2:
            dist_type = DistributionType.BIMODAL
        else:
            dist_type = DistributionType.MULTIMODAL

        # Compute confidence based on mode separation
        confidence = _compute_mode_separation_confidence(best_gmm)

        return DistributionAnalysisResult(
            distribution_type=dist_type,
            num_modes=best_n_components,
            global_statistics=global_stats,
            mode_statistics=mode_stats,
            mode_assignments=labels.tolist(),
            confidence=confidence,
        )

    except ImportError:
        logger.warning("sklearn not installed, falling back to unimodal analysis")
        return DistributionAnalysisResult(
            distribution_type=DistributionType.UNIMODAL,
            num_modes=1,
            global_statistics=global_stats,
            mode_statistics=[
                ModeStatistics(
                    mode_id=0,
                    mean=mean,
                    std=std,
                    count=len(clean_series),
                    weight=1.0,
                )
            ],
            confidence=1.0,
        )
    except Exception as e:
        logger.warning(f"GMM fitting failed: {e}, falling back to unimodal analysis")
        return DistributionAnalysisResult(
            distribution_type=DistributionType.UNIMODAL,
            num_modes=1,
            global_statistics=global_stats,
            mode_statistics=[
                ModeStatistics(
                    mode_id=0,
                    mean=mean,
                    std=std,
                    count=len(clean_series),
                    weight=1.0,
                )
            ],
            confidence=1.0,
        )


def compute_mode_statistics(
    series: pd.Series, labels: np.ndarray, weights: np.ndarray
) -> list[ModeStatistics]:
    """Compute statistics for each mode based on GMM labels.

    Args:
        series: Original data series
        labels: Per-sample mode assignments from GMM
        weights: GMM component weights

    Returns:
        List of ModeStatistics for each mode
    """
    mode_stats = []
    values = series.to_numpy()

    for mode_id in range(len(weights)):
        mask = labels == mode_id
        mode_values = values[mask]

        if len(mode_values) > 0:
            mode_mean = float(np.mean(mode_values))
            mode_std = float(np.std(mode_values)) if len(mode_values) > 1 else 0.0
            mode_stats.append(
                ModeStatistics(
                    mode_id=mode_id,
                    mean=round(mode_mean, 2),
                    std=round(mode_std, 2),
                    count=int(np.sum(mask)),
                    weight=round(float(weights[mode_id]), 3),
                )
            )

    # Sort by mean value for consistent ordering
    mode_stats.sort(key=lambda x: x.mean)

    return mode_stats


def _compute_mode_separation_confidence(gmm) -> float:
    """Compute confidence score based on how well-separated the modes are.

    Higher values mean modes are more distinct (less overlap).

    Args:
        gmm: Fitted GaussianMixture model

    Returns:
        Confidence score between 0 and 1
    """
    if gmm.n_components == 1:
        return 1.0

    means = gmm.means_.flatten()
    # Covariances can be various shapes depending on covariance_type
    if gmm.covariance_type == "full":
        stds = np.sqrt(gmm.covariances_[:, 0, 0])
    elif gmm.covariance_type == "tied":
        stds = np.sqrt(np.full(gmm.n_components, gmm.covariances_[0, 0]))
    elif gmm.covariance_type == "diag":
        stds = np.sqrt(gmm.covariances_[:, 0])
    else:  # spherical
        stds = np.sqrt(gmm.covariances_)

    # Compute pairwise separation ratios
    separations = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            distance = abs(means[i] - means[j])
            avg_std = (stds[i] + stds[j]) / 2
            if avg_std > 0:
                # Separation ratio: how many standard deviations apart
                separation = distance / (2 * avg_std)
                separations.append(separation)

    if not separations:
        return 0.0

    # Convert separation to confidence (sigmoid-like transformation)
    avg_separation = np.mean(separations)
    # >2 std separation = high confidence, <1 std = low confidence
    confidence = 1.0 / (1.0 + np.exp(-2 * (avg_separation - 1.5)))

    return round(float(confidence), 3)


def compute_multimodal_outliers(
    series: pd.Series,
    dist_result: DistributionAnalysisResult,
    std_threshold: float = 3.0,
) -> tuple[np.ndarray, dict]:
    """Detect outliers within each mode of a multimodal distribution.

    For unimodal distributions, uses standard 3-sigma detection.
    For multimodal distributions, detects outliers within each mode separately.

    Args:
        series: Data series to analyze
        dist_result: Distribution analysis result from detect_distribution_type
        std_threshold: Number of standard deviations for outlier detection

    Returns:
        Tuple of (outlier_mask, details_dict)
        - outlier_mask: Boolean array where True indicates an outlier
        - details_dict: Dictionary with outlier detection details
    """
    clean_series = series.dropna()
    n_samples = len(clean_series)

    if n_samples == 0:
        return np.array([], dtype=bool), {"num_outliers": 0}

    values = clean_series.to_numpy()
    outlier_mask = np.zeros(n_samples, dtype=bool)
    details: dict = {
        "distribution_type": dist_result.distribution_type.value,
        "num_modes": dist_result.num_modes,
        "outliers_per_mode": {},
    }

    if (
        dist_result.distribution_type == DistributionType.UNIMODAL
        or dist_result.mode_assignments is None
    ):
        # Standard 3-sigma outlier detection
        mean = dist_result.global_statistics["mean"]
        std = dist_result.global_statistics["std"]

        if std > 0:
            z_scores = np.abs((values - mean) / std)
            outlier_mask = z_scores > std_threshold

        details["outliers_per_mode"][0] = int(np.sum(outlier_mask))
    else:
        # Per-mode outlier detection
        labels = np.array(dist_result.mode_assignments)

        for mode_stat in dist_result.mode_statistics:
            mode_id = mode_stat.mode_id
            mode_mask = labels == mode_id
            mode_values = values[mode_mask]

            if len(mode_values) > 1 and mode_stat.std > 0:
                z_scores = np.abs((mode_values - mode_stat.mean) / mode_stat.std)
                mode_outliers = z_scores > std_threshold

                # Map back to original indices
                mode_indices = np.where(mode_mask)[0]
                outlier_mask[mode_indices[mode_outliers]] = True

                details["outliers_per_mode"][mode_id] = int(np.sum(mode_outliers))

    details["num_outliers"] = int(np.sum(outlier_mask))
    details["outlier_ratio"] = round(np.sum(outlier_mask) / n_samples, 4)

    return outlier_mask, details


def compute_statistics_with_distribution(
    series: pd.Series,
    decimal_precision: int = 2,
    include_distribution: bool = True,
) -> dict[str, Any]:
    """Compute statistics including distribution type detection.

    Enhanced version of compute_statistics that also detects multimodal
    distributions and provides per-mode statistics.

    Args:
        series: Pandas Series containing numeric values
        decimal_precision: Number of decimal places for rounding
        include_distribution: Whether to include distribution analysis

    Returns:
        Dictionary with statistics and distribution information
    """
    base_stats = compute_statistics(series, decimal_precision)

    if not include_distribution or series.empty or len(series) < 2:
        return base_stats

    # Detect distribution type
    dist_result = detect_distribution_type(series)

    # Add distribution information to stats
    base_stats["distribution_type"] = dist_result.distribution_type.value
    base_stats["num_modes"] = dist_result.num_modes
    base_stats["mode_separation_confidence"] = dist_result.confidence

    # Add per-mode statistics
    if dist_result.mode_statistics:
        base_stats["mode_statistics"] = [
            {
                "mode_id": ms.mode_id,
                "mean": ms.mean,
                "std": ms.std,
                "count": ms.count,
                "weight": ms.weight,
            }
            for ms in dist_result.mode_statistics
        ]

    return base_stats


def conversation_to_dataframes(
    conversation: Conversation,
    conversation_id: str,
    conversation_idx: int,
    chat_template: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a single conversation to separate conversation and message DataFrames.

    This creates two DataFrames: one for conversation-level data and one for
    message-level data, suitable for comprehensive dataset analysis.

    Args:
        conversation: The conversation object to convert
        conversation_id: ID of the conversation
        conversation_idx: Index of the conversation
        chat_template: Optional chat template name for formatting the conversation text.
            If None, uses simple 'ROLE: content' format.

    Returns:
        Tuple of (conversation_df, message_df)
    """
    # Create conversation-level data
    conversation_data = {
        "conversation_index": conversation_idx,
        "conversation_id": conversation_id,
        "num_messages": len(conversation.messages),
        "conversation_text_content": render_conversation_as_text(
            conversation, template_name=chat_template
        ),
    }

    if conversation.metadata:
        conversation_data.update(conversation.metadata)

    conversation_df = pd.DataFrame([conversation_data])

    # Create message-level data
    messages_data = []
    for msg_idx, message in enumerate(conversation.messages):
        text_content = (
            message.content
            if isinstance(message.content, str)
            else message.compute_flattened_text_content()
        )
        messages_data.append(
            {
                "conversation_index": conversation_idx,
                "conversation_id": conversation_id,
                "message_index": msg_idx,
                "message_id": message.id or f"msg_{msg_idx}",
                "role": message.role.value,
                "text_content": text_content,
            }
        )

    message_df = pd.DataFrame(messages_data)
    return conversation_df, message_df


def convert_dataset_to_dataframes(
    dataset,  # Union[BaseMapDataset, BaseIterableDataset]
    items_to_analyze: int,
    dataset_name: str = "Dataset",
    chat_template: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert a dataset to conversations and messages DataFrames.

    This method handles different dataset types:
    - SFT/Vision-SFT/GRPO datasets: Convert via conversation() method
    - DPO/KTO/Pretraining datasets: Convert raw data structure
    - Iterable datasets: Stream and limit to items_to_analyze

    Args:
        dataset: The dataset to process (BaseMapDataset or BaseIterableDataset)
        items_to_analyze: Number of items to analyze
        dataset_name: Name of the dataset for progress display
        chat_template: Optional chat template name for conversation formatting.
            If provided, conversations will be formatted using the specified template.

    Returns:
        Tuple of (conversations_df, messages_df) ready for analysis

    Raises:
        ValueError: If dataset is not provided
    """
    if dataset is None:
        raise ValueError("Dataset must be provided for conversation processing")

    # Handle iterable datasets (streaming datasets like C4)
    if isinstance(dataset, BaseIterableDataset):
        return _convert_iterable_dataset_to_dataframes(
            dataset, items_to_analyze, dataset_name
        )

    # Check if dataset has conversation() method (SFT/Vision-SFT/GRPO datasets)
    elif hasattr(dataset, "conversation") and callable(
        getattr(dataset, "conversation")
    ):
        return _convert_conversation_dataset_to_dataframes(
            dataset, items_to_analyze, dataset_name, chat_template
        )
    else:
        # For non-conversation datasets (DPO, KTO, pretraining), convert raw data
        return _convert_raw_dataset_to_dataframes(
            dataset, items_to_analyze, dataset_name
        )


def _convert_iterable_dataset_to_dataframes(
    dataset: BaseIterableDataset, items_to_analyze: int, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert iterable datasets (streaming) to DataFrames.

    This handles datasets like C4, Pile, etc. that are streamed and don't support index.
    We iterate through the dataset and stop after items_to_analyze items.
    """
    from oumi.utils.logging import logger

    # For now, create empty conversations DataFrame since these datasets don't have
    # conversation structure
    conversations_df = pd.DataFrame()

    # Stream raw data and limit to items_to_analyze
    raw_data_list = []

    try:
        # Use tqdm to show progress while streaming
        dataset_iter = iter(dataset)
        for idx in tqdm(
            range(items_to_analyze),
            desc=f"Streaming {dataset_name} data",
            unit="item",
        ):
            try:
                # Get next item from the iterator
                raw_item = next(dataset_iter)

                # Convert to dict if it's not already
                if hasattr(raw_item, "to_dict") and callable(
                    getattr(raw_item, "to_dict")
                ):
                    raw_item = raw_item.to_dict()  # type: ignore
                elif not isinstance(raw_item, dict):
                    # For pretraining datasets, the item might be a tensor or list
                    # Convert to a simple dict structure
                    raw_item = {"input_ids": raw_item, "item_index": idx}

                # Add index information
                raw_item["item_index"] = idx
                raw_data_list.append(raw_item)

            except StopIteration:
                # Dataset ended before we reached items_to_analyze
                logger.info(
                    f"Dataset ended after {idx} items (requested {items_to_analyze})"
                )
                break
            except Exception as e:
                logger.warning(f"Failed to process item {idx} from {dataset_name}: {e}")
                continue

    except Exception as e:
        logger.error(f"Failed to iterate over dataset {dataset_name}: {e}")
        # Return empty DataFrames if we can't iterate
        return pd.DataFrame(), pd.DataFrame()

    # Create a DataFrame from raw data for analysis
    if raw_data_list:
        messages_df = pd.DataFrame(raw_data_list)
        # Add required columns for analysis compatibility
        messages_df["conversation_index"] = messages_df["item_index"]
        messages_df["message_index"] = (
            0  # Single message per item for non-conversation data
        )
        messages_df["message_id"] = messages_df["item_index"].apply(
            lambda x: f"msg_{x}"
        )
    else:
        messages_df = pd.DataFrame()

    return conversations_df, messages_df


def _convert_conversation_dataset_to_dataframes(
    dataset,
    items_to_analyze: int,
    dataset_name: str,
    chat_template: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert datasets with conversation() method to DataFrames.

    Args:
        dataset: Dataset with conversation() method
        items_to_analyze: Number of items to analyze
        dataset_name: Name of the dataset for progress display
        chat_template: Optional chat template name for conversation formatting
    """
    conversation_df_list = []
    message_df_list = []

    for conversation_idx in tqdm(
        range(items_to_analyze),
        desc=f"Converting {dataset_name} to DataFrames",
        unit="item",
    ):
        conversation = dataset.conversation(conversation_idx)
        conversation_id = conversation.conversation_id or str(conversation_idx)
        conversation_df, message_df = conversation_to_dataframes(
            conversation, conversation_id, conversation_idx, chat_template
        )

        # Collect all DataFrames for concatenation
        if not conversation_df.empty:
            conversation_df_list.append(conversation_df)
        if not message_df.empty:
            message_df_list.append(message_df)

    # Create complete DataFrames by concatenating all individual DataFrames
    conversations_df = (
        pd.concat(conversation_df_list, ignore_index=True)
        if conversation_df_list
        else pd.DataFrame()
    )
    messages_df = (
        pd.concat(message_df_list, ignore_index=True)
        if message_df_list
        else pd.DataFrame()
    )

    return conversations_df, messages_df


def _convert_raw_dataset_to_dataframes(
    dataset, items_to_analyze: int, dataset_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert datasets without conversation() method to DataFrames.

    This handles DPO, KTO, and pretraining datasets that maintain their original.
    """
    from oumi.utils.logging import logger

    # For now, create empty conversations DataFrame since these datasets don't have
    # conversation structure
    conversations_df = pd.DataFrame()

    # Get raw data structure for analysis
    raw_data_list = []
    for idx in tqdm(
        range(items_to_analyze),
        desc=f"Converting {dataset_name} raw data",
        unit="item",
    ):
        try:
            # Get raw data from the dataset
            if hasattr(dataset, "raw"):
                raw_item = dataset.raw(idx)
            else:
                raw_item = dataset[idx]

            # Convert to dict if it's a pandas Series
            if hasattr(raw_item, "to_dict"):
                raw_item = raw_item.to_dict()

            # Add index information
            raw_item["item_index"] = idx
            raw_data_list.append(raw_item)

        except Exception as e:
            logger.warning(f"Failed to process item {idx} from {dataset_name}: {e}")
            continue

    # Create a DataFrame from raw data for analysis
    if raw_data_list:
        messages_df = pd.DataFrame(raw_data_list)
        # Add required columns for analysis compatibility
        messages_df["conversation_index"] = messages_df["item_index"]
        messages_df["message_index"] = (
            0  # Single message per item for non-conversation data
        )
        messages_df["message_id"] = messages_df["item_index"].apply(
            lambda x: f"msg_{x}"
        )
    else:
        messages_df = pd.DataFrame()

    return conversations_df, messages_df


def get_schema_for_format(dataset_format: str) -> dict:
    """Get column schema configuration based on dataset format.

    Args:
        dataset_format: Format of the dataset. Supported formats:
            - 'oumi': Conversation format (messages with roles)
            - 'alpaca': Instruction format (instruction/input/output)
            - 'prompt_response': Simple prompt/response pairs
            - 'dpo': Preference tuning format (prompt/chosen/rejected)
            - 'pretraining': Raw text for pretraining
            - 'kto': Binary feedback format (prompt/completion/label)

    Returns:
        Dictionary mapping column names to their configuration

    Raises:
        ValueError: If dataset_format is not supported
    """
    format_map = {
        "oumi": get_conversation_schema,
        "alpaca": get_alpaca_schema,
        "prompt_response": get_prompt_response_schema,
        "dpo": get_dpo_schema,
        "pretraining": get_pretraining_schema,
        "kto": get_kto_schema,
    }

    if dataset_format in format_map:
        return format_map[dataset_format]()
    else:
        raise ValueError(
            f"Unsupported dataset format: {dataset_format}. "
            f"Supported formats: {list(format_map.keys())}"
        )


def get_conversation_schema() -> dict:
    """Get column configuration for conversation format (oumi format).

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        # Conversation DataFrame columns
        "conversation_index": {
            "type": ColumnType.INT,
            "content_type": ContentType.IDENTIFIER,
            "description": "Conversation index in dataset",
        },
        "conversation_id": {
            "type": ColumnType.STRING,
            "content_type": ContentType.IDENTIFIER,
            "description": "Conversation identifier",
        },
        "num_messages": {
            "type": ColumnType.INT,
            "content_type": ContentType.NUMERIC,
            "description": "Number of messages in conversation",
        },
        "conversation_text_content": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Full conversation rendered as text",
        },
        # Message DataFrame columns
        "message_index": {
            "type": ColumnType.INT,
            "content_type": ContentType.IDENTIFIER,
            "description": "Message index within conversation",
        },
        "message_id": {
            "type": ColumnType.STRING,
            "content_type": ContentType.IDENTIFIER,
            "description": "Message identifier",
        },
        "role": {
            "type": ColumnType.STRING,
            "content_type": ContentType.CATEGORICAL,
            "description": "Message role (user/assistant/system)",
        },
        "text_content": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Message text content",
        },
    }


def get_alpaca_schema() -> dict:
    """Get column configuration for alpaca format (instruction format).

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "instruction": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Instruction or prompt text",
        },
        "input": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input context or additional information",
        },
        "output": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Expected output or response",
        },
        # Common additional fields
        "text": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Combined text content",
        },
    }


def get_prompt_response_schema() -> dict:
    """Get column configuration for prompt/response format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "prompt": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input prompt or question",
        },
        "response": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Model response or answer",
        },
        # Common variations
        "instruction": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Instruction or prompt text",
        },
        "output": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Expected output or response",
        },
    }


def get_dpo_schema() -> dict:
    """Get column configuration for DPO (preference tuning) format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "prompt": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input prompt for preference comparison",
        },
        "chosen": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Preferred response",
        },
        "rejected": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Rejected response",
        },
        # Metadata fields
        "score_chosen": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Score for chosen response",
        },
        "score_rejected": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Score for rejected response",
        },
    }


def get_pretraining_schema() -> dict:
    """Get column configuration for pretraining format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "text": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Raw text content for pretraining",
        },
        "content": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Text content (alternative field name)",
        },
    }


def get_kto_schema() -> dict:
    """Get column configuration for KTO (binary feedback) format.

    Returns:
        Dictionary mapping column names to their configuration.
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    return {
        "prompt": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Input prompt",
        },
        "completion": {
            "type": ColumnType.STRING,
            "content_type": ContentType.TEXT,
            "description": "Model completion/response",
        },
        "label": {
            "type": ColumnType.BOOL,
            "content_type": ContentType.CATEGORICAL,
            "description": "Binary feedback label (True=desirable, False=undesirable)",
        },
        # Additional fields
        "score": {
            "type": ColumnType.FLOAT,
            "content_type": ContentType.NUMERIC,
            "description": "Numeric score for the completion",
        },
    }


def augment_schema_with_dataframe_columns(schema: dict, df: pd.DataFrame) -> dict:
    """Augment schema with entries for columns not already in the schema.

    This function detects columns in the DataFrame that don't have schema
    entries (e.g., dynamic metadata columns) and adds default schema entries
    for them. This is useful for handling metadata fields that vary by dataset.

    Args:
        schema: Base schema dictionary
        df: DataFrame that may contain additional columns

    Returns:
        Updated schema dictionary with entries for all DataFrame columns
    """
    from oumi.core.analyze.column_types import ColumnType, ContentType

    augmented_schema = schema.copy()

    # Find columns in DataFrame that aren't in the schema
    missing_columns = set(df.columns) - set(schema.keys())

    if missing_columns:
        logger.info(
            f"Adding default schema entries for {len(missing_columns)} "
            f"columns not in base schema: {sorted(missing_columns)}"
        )

        # Add default schema entries for missing columns
        for col in missing_columns:
            # Infer type from the column data
            col_data = df[col]

            # Try to infer a reasonable type
            if pd.api.types.is_integer_dtype(col_data):
                col_type = ColumnType.INT
                content_type = ContentType.NUMERIC
            elif pd.api.types.is_float_dtype(col_data):
                col_type = ColumnType.FLOAT
                content_type = ContentType.NUMERIC
            elif pd.api.types.is_bool_dtype(col_data):
                col_type = ColumnType.BOOL
                content_type = ContentType.CATEGORICAL
            else:
                # Default to STRING/CATEGORICAL for unknown types
                # This is appropriate for metadata fields
                col_type = ColumnType.STRING
                content_type = ContentType.CATEGORICAL

            augmented_schema[col] = {
                "type": col_type,
                "content_type": content_type,
                "description": f"Additional field: {col}",
            }

    return augmented_schema


def save_analyzer_artifacts(
    analyzer: Any,  # DatasetAnalyzer
    output_path: str | Path,
    output_format: str = "parquet",
) -> None:
    """Save all analyzer artifacts to disk.

    This function saves all analysis results including dataframes, schemas, and
    summary. The output format can be 'csv', 'json', or 'parquet' for dataframes.
    Schemas and summary are always saved as JSON.

    Args:
        analyzer: The DatasetAnalyzer instance with completed analysis
        output_path: Directory path where artifacts will be saved
        output_format: Format for dataframes ('csv', 'json', or 'parquet').
            Defaults to 'parquet'. Case-insensitive.

    Raises:
        RuntimeError: If analysis has not been run yet.
        ValueError: If output_format is invalid.
    """
    import json

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_format = output_format.lower()
    valid_formats = {"csv", "json", "parquet"}
    if output_format not in valid_formats:
        raise ValueError(
            f"Invalid output format '{output_format}'. "
            f"Supported formats: {', '.join(valid_formats)}"
        )

    # Check if analysis has been run
    if analyzer._merged_df is None:
        raise RuntimeError(
            "Analysis has not been run yet. Please call analyze_dataset() first "
            "before saving artifacts."
        )

    def _save_dataframe(df: pd.DataFrame, path: Path, fmt: str) -> None:
        """Save a DataFrame to the specified format."""
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", indent=2)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)

    # Save message-level results
    if analyzer.message_df is not None and not analyzer.message_df.empty:
        msg_path = output_dir / f"messages_df.{output_format}"
        _save_dataframe(analyzer.message_df, msg_path, output_format)
        logger.info(f"Saved message analysis to: {msg_path}")

    # Save conversation-level results
    if analyzer.conversation_df is not None and not analyzer.conversation_df.empty:
        conv_path = output_dir / f"conversations_df.{output_format}"
        _save_dataframe(analyzer.conversation_df, conv_path, output_format)
        logger.info(f"Saved conversation analysis to: {conv_path}")

    # Save merged results
    if analyzer.analysis_df is not None and not analyzer.analysis_df.empty:
        merged_path = output_dir / f"merged_df.{output_format}"
        _save_dataframe(analyzer.analysis_df, merged_path, output_format)
        logger.info(f"Saved merged analysis to: {merged_path}")

    # Save schemas - both combined and individual files
    schemas = {}
    if analyzer._merged_schema is not None:
        schemas["merged_schema"] = analyzer._merged_schema
    if analyzer._message_schema is not None:
        schemas["message_schema"] = analyzer._message_schema
        # Also save message schema as separate file
        message_schema_path = output_dir / "message_schema.json"
        with open(message_schema_path, "w") as f:
            json.dump(analyzer._message_schema, f, indent=2, default=str)
        logger.info(f"Saved message schema to: {message_schema_path}")
    if analyzer._conversation_schema is not None:
        schemas["conversation_schema"] = analyzer._conversation_schema
        # Also save conversation schema as separate file
        conversation_schema_path = output_dir / "conversation_schema.json"
        with open(conversation_schema_path, "w") as f:
            json.dump(analyzer._conversation_schema, f, indent=2, default=str)
        logger.info(f"Saved conversation schema to: {conversation_schema_path}")

    if schemas:
        schema_path = output_dir / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(schemas, f, indent=2, default=str)
        logger.info(f"Saved combined schemas to: {schema_path}")

    # Save analysis summary
    if analyzer._analysis_summary is not None:
        summary_path = output_dir / "analysis_summary.json"
        with open(summary_path, "w") as f:
            json.dump(analyzer.analysis_summary, f, indent=2, default=str)
        logger.info(f"Saved analysis summary to: {summary_path}")

    logger.info(f"All analyzer artifacts saved to: {output_dir.absolute()}")


def load_analyzer_artifacts(
    input_path: str | Path,
    output_format: str = "parquet",
) -> dict[str, Any]:
    """Load analyzer artifacts from disk.

    This function loads all saved analysis results including dataframes, schemas,
    and summary. The function attempts to detect the format automatically if
    files exist, but you can specify the expected format.

    Args:
        input_path: Directory path where artifacts were saved
        output_format: Expected format for dataframes ('csv', 'json', or 'parquet').
            Defaults to 'parquet'. Case-insensitive. If files don't exist with this
            format, will try other formats.

    Returns:
        Dictionary containing loaded artifacts with keys:
        - 'messages_df': Message-level DataFrame (if available)
        - 'conversations_df': Conversation-level DataFrame (if available)
        - 'merged_df': Merged DataFrame (if available)
        - 'schemas': Dictionary with 'merged_schema', 'message_schema',
          'conversation_schema'
        - 'analysis_summary': Analysis summary dictionary (if available)

    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If output_format is invalid.
    """
    import json

    input_dir = Path(input_path)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise ValueError(f"Input path must be a directory: {input_dir}")

    output_format = output_format.lower()
    valid_formats = {"csv", "json", "parquet"}
    if output_format not in valid_formats:
        raise ValueError(
            f"Invalid output format '{output_format}'. "
            f"Supported formats: {', '.join(valid_formats)}"
        )

    artifacts: dict[str, Any] = {}

    def _load_dataframe(path: Path, fmt: str) -> pd.DataFrame | None:
        """Load DataFrame from specified format, trying multiple if needed."""
        # Try the specified format first
        formats_to_try = [fmt] + [f for f in valid_formats if f != fmt]

        for fmt_to_try in formats_to_try:
            file_path = path.with_suffix(f".{fmt_to_try}")
            if file_path.exists():
                try:
                    if fmt_to_try == "csv":
                        return pd.read_csv(file_path)
                    elif fmt_to_try == "json":
                        return pd.read_json(file_path, orient="records")
                    elif fmt_to_try == "parquet":
                        return pd.read_parquet(file_path)
                except Exception as e:
                    logger.warning(
                        f"Failed to load {file_path} as {fmt_to_try}: {e}. "
                        f"Trying other formats..."
                    )
                    continue
        return None

    # Load message-level results
    msg_path = input_dir / "messages_df"
    messages_df = _load_dataframe(msg_path, output_format)
    if messages_df is not None:
        artifacts["messages_df"] = messages_df
        logger.info(f"Loaded message analysis from: {msg_path}")

    # Load conversation-level results
    conv_path = input_dir / "conversations_df"
    conversations_df = _load_dataframe(conv_path, output_format)
    if conversations_df is not None:
        artifacts["conversations_df"] = conversations_df
        logger.info(f"Loaded conversation analysis from: {conv_path}")

    # Load merged results
    merged_path = input_dir / "merged_df"
    merged_df = _load_dataframe(merged_path, output_format)
    if merged_df is not None:
        artifacts["merged_df"] = merged_df
        logger.info(f"Loaded merged analysis from: {merged_path}")

    # Load schemas - try combined file first, then individual files
    schema_path = input_dir / "schema.json"
    schemas = {}

    if schema_path.exists():
        with open(schema_path) as f:
            schemas = json.load(f)
        logger.info(f"Loaded combined schemas from: {schema_path}")
    else:
        # Try loading individual schema files
        message_schema_path = input_dir / "message_schema.json"
        if message_schema_path.exists():
            with open(message_schema_path) as f:
                schemas["message_schema"] = json.load(f)
            logger.info(f"Loaded message schema from: {message_schema_path}")

        conversation_schema_path = input_dir / "conversation_schema.json"
        if conversation_schema_path.exists():
            with open(conversation_schema_path) as f:
                schemas["conversation_schema"] = json.load(f)
            logger.info(f"Loaded conversation schema from: {conversation_schema_path}")

    artifacts["schemas"] = schemas

    # Load analysis summary
    summary_path = input_dir / "analysis_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            artifacts["analysis_summary"] = json.load(f)
        logger.info(f"Loaded analysis summary from: {summary_path}")

    logger.info(f"Loaded analyzer artifacts from: {input_dir.absolute()}")
    return artifacts


def regenerate_recommendations(
    artifacts: dict[str, Any],
    outlier_threshold: float = 3.0,
) -> dict[str, Any]:
    """Regenerate recommendations from loaded artifacts using the latest code.

    This function regenerates recommendations from saved artifacts, ensuring
    they use the latest recommendation logic (e.g., updated duplicate detection).

    Args:
        artifacts: Dictionary of loaded artifacts from load_analyzer_artifacts().
            Must include 'messages_df', 'conversations_df', and 'analysis_summary'.
        outlier_threshold: Standard deviation threshold for outlier detection.
            Defaults to 3.0.

    Returns:
        Updated artifacts dictionary with regenerated recommendations in
        artifacts['analysis_summary']['recommendations'].

    Raises:
        ValueError: If required artifacts are missing.
    """
    from oumi.core.analyze.recommendations import RecommendationsEngine

    message_df = artifacts.get("messages_df")
    conversation_df = artifacts.get("conversations_df")
    summary = artifacts.get("analysis_summary")

    if message_df is None:
        raise ValueError("artifacts must include 'messages_df'")
    if summary is None:
        raise ValueError("artifacts must include 'analysis_summary'")

    if message_df.empty:
        logger.warning("messages_df is empty, skipping recommendation regeneration")
        return artifacts

    # Handle conversation_df properly (avoid DataFrame boolean ambiguity)
    conv_df = conversation_df if conversation_df is not None else pd.DataFrame()

    engine = RecommendationsEngine(outlier_std_threshold=outlier_threshold)
    recommendations = engine.generate_recommendations(
        message_df=message_df,
        conversation_df=conv_df,
        analysis_summary=summary,
    )

    # Update artifacts with regenerated recommendations
    regenerated_recs = [rec.to_dict() for rec in recommendations]
    summary["recommendations"] = regenerated_recs
    artifacts["analysis_summary"]["recommendations"] = regenerated_recs

    logger.info(
        f"Regenerated {len(recommendations)} recommendations "
        "from artifacts with latest code"
    )

    # Debug: Log if duplicate recommendations were regenerated
    duplicate_recs = [r for r in recommendations if "duplicate" in r.title.lower()]
    if duplicate_recs:
        logger.info(
            f"Found {len(duplicate_recs)} duplicate-related "
            f"recommendations after regeneration"
        )

    return artifacts
