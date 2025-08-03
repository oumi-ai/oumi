#!/usr/bin/env python3
"""Script to synthesize images for dataset samples using local VLLM inference via Oumi.

Accepts a HuggingFace dataset and generates code artifacts to create appropriate images.
Supports local multi-GPU inference with text-only models (Llama, Qwen, etc.).
Supports resuming from checkpoints and intermittent saving of results.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference.vllm_inference_engine import VLLMInferenceEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """Get GPU memory information for monitoring."""
    if not torch.cuda.is_available():
        return None

    gpu_info = []
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        free = total - cached
        gpu_info.append(
            {
                "device": i,
                "total": total,
                "allocated": allocated,
                "cached": cached,
                "free": free,
            }
        )
    return gpu_info


def log_gpu_memory_usage():
    """Log current GPU memory usage."""
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        for info in gpu_info:
            logger.info(
                f"GPU {info['device']}: {info['allocated']:.1f}GB allocated, "
                f"{info['free']:.1f}GB free, {info['total']:.1f}GB total"
            )


def optimize_batch_size_for_model(model_name: str, default_batch_size: int) -> int:
    """Optimize batch size based on model size and available GPU memory."""
    # Simple heuristics based on common model sizes
    model_name_lower = model_name.lower()

    if "405b" in model_name_lower:
        return max(1, default_batch_size // 8)  # Very large models
    elif "70b" in model_name_lower or "72b" in model_name_lower:
        return max(1, default_batch_size // 4)  # Large models
    elif "13b" in model_name_lower or "14b" in model_name_lower:
        return max(2, default_batch_size // 2)  # Medium-large models
    elif "7b" in model_name_lower or "8b" in model_name_lower:
        return default_batch_size  # Medium models
    else:
        return min(
            default_batch_size * 2, 100
        )  # Small models can handle larger batches

    return default_batch_size


class CheckpointManager:
    """Manages checkpoints for resuming interrupted jobs."""

    def __init__(self, output_path: str):
        """Initialize checkpoint manager."""
        self.output_path = Path(output_path)
        self.checkpoint_file = self.output_path / "checkpoint.json"
        self.temp_results_file = self.output_path / "temp_results.json"
        self.output_path.mkdir(exist_ok=True, parents=True)

    def load_checkpoint(self) -> tuple[int, list[Optional[str]]]:
        """Load checkpoint if it exists. Returns (start_index, existing_artifacts)."""
        if not self.checkpoint_file.exists():
            return 0, []

        try:
            with open(self.checkpoint_file) as f:
                checkpoint = json.load(f)

            start_index = checkpoint.get("last_processed_index", -1) + 1
            artifacts = checkpoint.get("artifacts", [])
            logger.info(
                f"Resuming from checkpoint: {start_index} samples already processed"
            )
            return start_index, artifacts
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Starting from beginning.")
            return 0, []

    def save_checkpoint(self, index: int, artifacts: list[Optional[str]]):
        """Save current progress to checkpoint file."""
        checkpoint = {
            "last_processed_index": index,
            "artifacts": artifacts,
            "timestamp": time.time(),
        }

        # Write to temporary file first, then rename for atomicity
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.rename(self.checkpoint_file)

        logger.debug(f"Checkpoint saved at index {index}")

    def save_temp_results(
        self, dataset_dict: dict[str, Any], artifacts: list[Optional[str]]
    ):
        """Save temporary results that can be used to create the final dataset."""
        temp_data = []
        for i, (key, sample) in enumerate(dataset_dict.items()):
            sample_data = self._make_json_serializable(
                sample.copy() if isinstance(sample, dict) else {"data": sample}
            )
            if i < len(artifacts):
                sample_data["image_synthesis_code"] = artifacts[i]
            temp_data.append(sample_data)

        with open(self.temp_results_file, "w") as f:
            json.dump(temp_data, f, indent=2)

        logger.debug(f"Temporary results saved: {len(temp_data)} samples")

    def cleanup(self):
        """Remove checkpoint files after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.temp_results_file.exists():
            self.temp_results_file.unlink()
        logger.info("Checkpoint files cleaned up")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle custom objects by converting to dict
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj


def extract_question_from_sample(sample: dict[str, Any]) -> str:
    """Extract question/text content from various dataset formats."""
    # Check for ShareGPT conversation format
    if "conversations" in sample:
        conversations = sample["conversations"]
        if isinstance(conversations, list):
            # Find the first human message
            for msg in conversations:
                if isinstance(msg, dict) and msg.get("from") == "human":
                    return msg.get("value", "")
            # Fallback: use first message if no human message found
            if conversations and isinstance(conversations[0], dict):
                return conversations[0].get("value", str(sample))

    # Check standard formats
    text_content = sample.get(
        "text", sample.get("prompt", sample.get("question", None))
    )

    if text_content:
        return text_content

    # Fallback: convert entire sample to string
    return str(sample)


def create_image_synthesis_prompt(sample: dict[str, Any]) -> str:
    """Create a prompt for Claude to generate image synthesis code."""
    # Extract the text content from the sample
    text_content = extract_question_from_sample(sample)

    prompt = f"""Given the following educational content from a dataset sample, create a
Python code artifact that will generate or download an appropriate image to accompany
this content.

DATASET SAMPLE:
{text_content}

REQUIREMENTS:
1. If the content involves mathematical concepts, formulas, or data relationships,
   create a matplotlib/seaborn visualization (graph, plot, diagram).
2. If the content would benefit from a real-world image (e.g., scientific phenomena,
   historical events, objects), write code to download an appropriate image from a
   reputable source.
3. The code must be completely self-contained and runnable without any additional setup.
4. The code should save the image as 'output_image.png' in the current directory.
5. Include all necessary imports and error handling.
6. For web-scraped images, use appropriate attribution and respect copyright.

IMPORTANT: Return ONLY a code artifact enclosed in ```python tags. The code should be
ready to run immediately.

Consider the educational value: What visual would best help someone understand this
concept?"""

    return prompt


def extract_code_artifact(response: str) -> Optional[str]:
    """Extract Python code artifact from Claude's response."""
    # Look for code blocks marked with ```python
    code_pattern = r"```python\n(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)

    if matches:
        # Return the first (and should be only) code block
        return matches[0].strip()

    # Fallback: look for any code block
    code_pattern_generic = r"```\n(.*?)```"
    matches = re.findall(code_pattern_generic, response, re.DOTALL)

    if matches:
        return matches[0].strip()

    logger.warning("No code artifact found in response")
    return None


def setup_vllm_engine(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    tensor_parallel_size: int = -1,
    gpu_memory_utilization: float = 0.9,
    enable_prefix_caching: bool = True,
):
    """Set up the VLLM inference engine via Oumi."""
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. VLLM requires GPU support.")

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        raise RuntimeError("No GPUs detected. VLLM requires at least one GPU.")

    # Set tensor parallel size
    if tensor_parallel_size == -1:
        tensor_parallel_size = gpu_count
    elif tensor_parallel_size > gpu_count:
        logger.warning(
            f"Requested tensor_parallel_size ({tensor_parallel_size}) > available GPUs "
            f"({gpu_count}). Using {gpu_count} GPUs."
        )
        tensor_parallel_size = gpu_count

    logger.info(f"Setting up VLLM engine with model: {model_name}")
    logger.info(f"Using {tensor_parallel_size}/{gpu_count} GPUs for tensor parallelism")
    logger.info(f"GPU memory utilization: {gpu_memory_utilization}")

    # Configure model parameters
    model_params = ModelParams(
        model_name=model_name,
        torch_dtype_str="bfloat16",  # Optimal for most modern GPUs
        trust_remote_code=True,
    )

    # Initialize the VLLM engine directly (builder doesn't support VLLM-specific params)
    engine = VLLMInferenceEngine(
        model_params=model_params,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=True,  # Recommended for stability
    )

    logger.info(f"VLLM engine created successfully: {type(engine)}")
    return engine


def create_conversation_from_sample(sample: dict[str, Any]) -> Conversation:
    """Create a conversation from a dataset sample."""
    prompt = create_image_synthesis_prompt(sample)
    return Conversation(messages=[Message(role=Role.USER, content=prompt)])


def process_dataset_batch(
    engine, samples: list[dict[str, Any]], generation_params: GenerationParams
) -> list[Optional[str]]:  # noqa: PLR0915
    """Process a batch of dataset samples and return the code artifacts."""
    try:
        # Create conversations for all samples
        conversations = [create_conversation_from_sample(sample) for sample in samples]

        # Create inference config with model parameters from engine
        inference_config = InferenceConfig(
            generation=generation_params,
            model=engine.model_params if hasattr(engine, "model_params") else None,
        )

        # Run batch inference
        logger.info(f"Running VLLM batch inference on {len(conversations)} samples")
        model_name = (
            engine.model_params.model_name
            if hasattr(engine, "model_params")
            else "Unknown"
        )
        logger.info(f"Model: {model_name}")
        results = engine.infer(conversations, inference_config)

        # Extract code artifacts from results
        code_artifacts = []
        if results and len(results) > 0:
            for i, result in enumerate(results):
                if result and result.messages:
                    response = result.messages[-1].content
                    code_artifact = extract_code_artifact(response)
                    code_artifacts.append(code_artifact)
                    if code_artifact:
                        logger.debug(
                            f"Sample {i}: Code artifact extracted successfully"
                        )
                    else:
                        logger.warning(
                            f"Sample {i}: No code artifact found in response"
                        )
                else:
                    logger.error(f"Sample {i}: No response received")
                    code_artifacts.append(None)
        else:
            logger.error("No results received from batch inference")
            code_artifacts = [None] * len(samples)

        return code_artifacts

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing batch: {error_msg}")

        # Provide helpful guidance for known VLLM/GPU issues
        if "CUDA" in error_msg or "GPU" in error_msg:
            logger.info("GPU/CUDA issue detected. Please check:")
            logger.info("1. CUDA is properly installed and compatible")
            logger.info("2. Sufficient GPU memory is available")
            logger.info("3. Model fits in available GPU memory")
            logger.info(
                "Consider reducing --gpu-memory-utilization or using a smaller model"
            )
        elif "out of memory" in error_msg.lower() or "oom" in error_msg.lower():
            logger.info("Out of memory error detected. Try:")
            logger.info("1. Reducing batch size")
            logger.info("2. Using --gpu-memory-utilization=0.7 or lower")
            logger.info("3. Using a smaller model")
            logger.info("4. Reducing --tensor-parallel-size")
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            logger.info("Model not found. Please verify:")
            logger.info("1. Model name is correct")
            logger.info("2. Model is available on HuggingFace")
            logger.info("3. You have access to the model (if gated)")

        return [None] * len(samples)


def get_output_path(input_dataset_name: str, output_path: Optional[str]) -> str:
    """Determine the output path for the dataset."""
    if output_path:
        return output_path

    # Create local equivalent of the input path
    # Convert HuggingFace dataset name to a valid directory name
    safe_name = input_dataset_name.replace("/", "_").replace(":", "_")
    return f"./synthesized_{safe_name}"


def main():  # noqa: PLR0915
    """Run the image synthesis pipeline."""
    parser = argparse.ArgumentParser(
        description="Synthesize images for dataset samples using local VLLM "
        "inference via Oumi"
    )
    parser.add_argument(
        "input",
        type=str,
        help="HuggingFace dataset identifier (e.g., 'simplescaling/s1K-1.1')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the synthesized dataset "
        "(defaults to local equivalent of input)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model name to use for inference "
        "(default: meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save checkpoint every N chunks when using batch processing, "
        "or every N samples otherwise (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of samples to process in each batch chunk "
        "(will be auto-optimized based on model size, default: 50)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=-1,
        help="Number of GPUs to use for tensor parallelism "
        "(-1 uses all available GPUs, default: -1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)",
    )

    args = parser.parse_args()

    # Determine output path first (needed for checkpoint manager)
    output_path = get_output_path(args.input, args.output)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(output_path)

    # Load checkpoint if resuming
    start_index = 0
    code_artifacts = []

    if args.resume:
        start_index, code_artifacts = checkpoint_manager.load_checkpoint()

    # Check initial GPU status
    logger.info("Checking GPU availability...")
    if torch.cuda.is_available():
        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        log_gpu_memory_usage()
    else:
        logger.error("No CUDA-capable GPUs found")
        return 1

    # Set up the VLLM engine
    logger.info("Setting up VLLM inference engine...")
    try:
        engine = setup_vllm_engine(
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
        )

        # Log GPU usage after model loading
        logger.info("GPU memory usage after model loading:")
        log_gpu_memory_usage()

    except Exception as e:
        logger.error(f"Failed to set up VLLM engine: {str(e)}")
        return 1

    # Load the dataset
    logger.info(f"Loading dataset: {args.input}")
    try:
        dataset = load_dataset(args.input, split=args.split)
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return 1

    # Set up generation parameters
    generation_params = GenerationParams(
        max_new_tokens=4096, temperature=args.temperature, top_p=0.95
    )

    # Optimize batch size based on model
    optimized_batch_size = optimize_batch_size_for_model(
        args.model_name, args.batch_size
    )
    if optimized_batch_size != args.batch_size:
        logger.info(
            f"Optimizing batch size from {args.batch_size} to {optimized_batch_size} "
            f"based on model size"
        )
        args.batch_size = optimized_batch_size

    # Process each sample
    total_samples = len(dataset)
    if start_index > 0:
        logger.info(f"Resuming from sample {start_index}/{total_samples}")
    else:
        logger.info(f"Processing {total_samples} samples...")

    # Ensure artifacts list has the right size
    while len(code_artifacts) < total_samples:
        code_artifacts.append(None)

    # Process remaining samples in chunks
    try:
        remaining_count = total_samples - start_index
        chunk_count = (
            remaining_count + args.batch_size - 1
        ) // args.batch_size  # Ceiling division

        logger.info(
            f"Processing {remaining_count} samples in {chunk_count} chunks of "
            f"{args.batch_size} samples each..."
        )

        # Process samples in chunks
        for chunk_idx in range(chunk_count):
            start_idx = start_index + (chunk_idx * args.batch_size)
            end_idx = min(start_idx + args.batch_size, total_samples)

            # Get samples for this chunk
            chunk_samples = [dataset[i] for i in range(start_idx, end_idx)]
            actual_chunk_size = len(chunk_samples)

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{chunk_count}: samples "
                f"{start_idx}-{end_idx - 1} ({actual_chunk_size} samples)"
            )

            # Process this chunk
            with tqdm(
                total=actual_chunk_size,
                desc=f"Chunk {chunk_idx + 1}/{chunk_count}",
                leave=False,
            ) as chunk_pbar:
                try:
                    chunk_artifacts = process_dataset_batch(
                        engine, chunk_samples, generation_params
                    )
                    chunk_pbar.update(actual_chunk_size)

                    # Log GPU memory usage periodically
                    if (chunk_idx + 1) % 5 == 0:
                        logger.info(f"GPU memory usage after chunk {chunk_idx + 1}:")
                        log_gpu_memory_usage()

                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        f"CUDA out of memory in chunk {chunk_idx + 1}. "
                        f"Trying to recover..."
                    )
                    torch.cuda.empty_cache()  # Clear cache

                    # Try with smaller batch
                    if actual_chunk_size > 1:
                        logger.info(
                            f"Retrying chunk {chunk_idx + 1} with smaller batch size..."
                        )
                        smaller_batches = [
                            chunk_samples[i : i + max(1, actual_chunk_size // 2)]
                            for i in range(
                                0, len(chunk_samples), max(1, actual_chunk_size // 2)
                            )
                        ]
                        chunk_artifacts = []
                        for small_batch in smaller_batches:
                            try:
                                small_results = process_dataset_batch(
                                    engine, small_batch, generation_params
                                )
                                chunk_artifacts.extend(small_results)
                            except Exception as e2:
                                logger.error(f"Failed even with smaller batch: {e2}")
                                chunk_artifacts.extend([None] * len(small_batch))
                    else:
                        logger.error(
                            f"Cannot reduce batch size further for chunk "
                            f"{chunk_idx + 1}"
                        )
                        chunk_artifacts = [None] * actual_chunk_size

                    chunk_pbar.update(actual_chunk_size)

                except Exception as e:
                    logger.error(f"Unexpected error in chunk {chunk_idx + 1}: {e}")
                    chunk_artifacts = [None] * actual_chunk_size
                    chunk_pbar.update(actual_chunk_size)

            # Update the code_artifacts list with chunk results
            for i, artifact in enumerate(chunk_artifacts):
                code_artifacts[start_idx + i] = artifact

            # Save checkpoint at intervals or on last chunk
            if (
                chunk_idx + 1
            ) % args.save_interval == 0 or chunk_idx == chunk_count - 1:
                checkpoint_manager.save_checkpoint(
                    end_idx - 1, code_artifacts[:end_idx]
                )

                # Also save temporary results
                data_frame = dataset.select(range(end_idx)).to_pandas()
                dataset_dict = {}
                for idx, row in data_frame.iterrows():
                    dataset_dict[str(idx)] = _convert_to_json_serializable(row)
                checkpoint_manager.save_temp_results(
                    dataset_dict, code_artifacts[:end_idx]
                )
                logger.info(
                    f"Progress saved: {end_idx}/{total_samples} samples processed "
                    f"({chunk_idx + 1}/{chunk_count} chunks)"
                )

        logger.info(
            f"Batch processing complete: {total_samples}/{total_samples} samples "
            f"processed in {chunk_count} chunks"
        )

    except KeyboardInterrupt:
        logger.warning("Process interrupted! Saving checkpoint...")
        # Save checkpoint for the last completed samples
        last_completed = len([a for a in code_artifacts if a is not None])
        if last_completed > 0:
            checkpoint_manager.save_checkpoint(
                last_completed - 1, code_artifacts[:last_completed]
            )
        raise
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        # Save checkpoint for the last completed samples
        last_completed = len([a for a in code_artifacts if a is not None])
        if last_completed > 0:
            checkpoint_manager.save_checkpoint(
                last_completed - 1, code_artifacts[:last_completed]
            )
        raise

    # Add the code artifacts to the dataset
    logger.info("Adding code artifacts to dataset...")

    # Convert dataset to pandas for easier manipulation
    data_frame = dataset.to_pandas()
    data_frame["image_synthesis_code"] = code_artifacts[: len(data_frame)]

    # Convert back to dataset
    enhanced_dataset = Dataset.from_pandas(data_frame)

    # Save the enhanced dataset
    logger.info(f"Saving enhanced dataset to: {output_path}")
    enhanced_dataset.save_to_disk(output_path)

    # Also save as JSON for easy inspection
    json_path = Path(output_path) / "dataset.json"
    enhanced_dataset.to_json(str(json_path))

    # Clean up checkpoint files on success
    checkpoint_manager.cleanup()

    # Final GPU memory check
    logger.info("Final GPU memory usage:")
    log_gpu_memory_usage()

    # Summary statistics
    successful_artifacts = sum(1 for artifact in code_artifacts if artifact is not None)
    logger.info(
        f"Successfully generated {successful_artifacts}/{len(dataset)} code artifacts"
    )
    logger.info(f"Success rate: {successful_artifacts / len(dataset) * 100:.1f}%")
    logger.info(f"Dataset saved to: {output_path}")

    if successful_artifacts < len(dataset):
        failed_count = len(dataset) - successful_artifacts
        logger.warning(
            f"{failed_count} samples failed to generate code artifacts. "
            f"Check logs for details."
        )

    return 0


def _convert_to_json_serializable(obj: Any) -> Any:
    """Convert objects to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Handle custom objects by converting to dict
        return _convert_to_json_serializable(obj.__dict__)
    else:
        return obj


def test_vllm_setup(model_name: str = "meta-llama/Llama-3.2-3B-Instruct") -> bool:
    """Test function to verify VLLM setup is working correctly."""
    logger.info(f"Testing VLLM setup with model: {model_name}")

    try:
        # Test GPU availability
        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            return False

        logger.info(f"Found {torch.cuda.device_count()} GPU(s)")
        log_gpu_memory_usage()

        # Test engine creation
        engine = setup_vllm_engine(model_name, tensor_parallel_size=1)
        logger.info("Engine created successfully")

        # Test simple inference
        test_conversation = Conversation(
            messages=[Message(role=Role.USER, content="Hello, how are you?")]
        )

        generation_params = GenerationParams(max_new_tokens=50, temperature=0.7)

        inference_config = InferenceConfig(
            generation=generation_params,
            model=engine.model_params if hasattr(engine, "model_params") else None,
        )

        results = engine.infer([test_conversation], inference_config)

        if results and len(results) > 0 and results[0].messages:
            response = results[0].messages[-1].content
            logger.info(f"Test inference successful. Response: {response[:100]}...")
            return True
        else:
            logger.error("No response received from test inference")
            return False

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import sys

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        model_name = (
            sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.2-3B-Instruct"
        )
        success = test_vllm_setup(model_name)
        sys.exit(0 if success else 1)
    else:
        sys.exit(main())
