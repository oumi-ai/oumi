import time
from pathlib import Path
from typing import NamedTuple

import pytest

from oumi import infer, infer_interactive
from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.utils.image_utils import load_image_png_bytes_from_path
from tests.integration.infer import get_default_device_map_for_inference
from tests.integration.infer.test_inference_test_utils import (
    assert_performance_requirements,
    assert_response_properties,
    assert_response_relevance,
    count_response_tokens,
    get_test_models,
    validate_generation_output,
)
from tests.markers import requires_cuda_initialized, requires_gpus

FIXED_PROMPT = "Hello world!"
FIXED_RESPONSE = "The U.S."


class InferTestSpec(NamedTuple):
    num_batches: int
    batch_size: int


def _get_infer_test_spec_id(x):
    assert isinstance(x, InferTestSpec)
    return f"batches={x.num_batches} bs={x.batch_size}"


def _compare_conversation_lists(
    output: list[Conversation],
    expected_output: list[Conversation],
) -> bool:
    if len(output) != len(expected_output):
        return False

    for actual, expected in zip(output, expected_output):
        if actual.messages != expected.messages:
            return False
        if actual.metadata != expected.metadata:
            return False
        if expected.conversation_id is not None:
            if actual.conversation_id != expected.conversation_id:
                return False

    return True


@requires_cuda_initialized()
@requires_gpus()
def test_infer_basic_interactive(monkeypatch: pytest.MonkeyPatch):
    models = get_test_models()
    model_params = models["smollm_135m"]

    config: InferenceConfig = InferenceConfig(
        model=model_params,
        generation=GenerationParams(max_new_tokens=10, temperature=0.0, seed=42),
    )

    # Simulate the user entering "Hello world!" in the terminal folowed by Ctrl+D.
    input_iterator = iter([FIXED_PROMPT])

    def mock_input(_):
        try:
            return next(input_iterator)
        except StopIteration:
            raise EOFError  # Simulate Ctrl+D

    # Replace the built-in input function
    monkeypatch.setattr("builtins.input", mock_input)
    infer_interactive(config)


@requires_cuda_initialized()
@requires_gpus()
def test_infer_basic_interactive_with_images(
    monkeypatch: pytest.MonkeyPatch, root_testdata_dir: Path
):
    config: InferenceConfig = InferenceConfig(
        model=ModelParams(
            model_name="HuggingFaceTB/SmolVLM-Instruct",
            model_max_length=1024,
            trust_remote_code=True,
            torch_dtype_str="bfloat16",  # Use bfloat16 for efficiency
            device_map="auto",
        ),
        generation=GenerationParams(max_new_tokens=10, temperature=0.0, seed=42),
    )

    png_image_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "the_great_wave_off_kanagawa.jpg"
    )

    # Simulate the user entering "Hello world!" in the terminal folowed by Ctrl+D.
    input_iterator = iter(["Describe the image!"])

    def mock_input(_):
        try:
            return next(input_iterator)
        except StopIteration:
            raise EOFError  # Simulate Ctrl+D

    # Replace the built-in input function
    monkeypatch.setattr("builtins.input", mock_input)
    infer_interactive(config, input_image_bytes=[png_image_bytes])


@pytest.mark.parametrize(
    "test_spec",
    [
        InferTestSpec(num_batches=1, batch_size=1),
        InferTestSpec(num_batches=1, batch_size=2),
        InferTestSpec(num_batches=2, batch_size=1),
    ],  # Reduced test cases for faster execution
    ids=_get_infer_test_spec_id,
)
def test_infer_basic_non_interactive(test_spec: InferTestSpec):
    models = get_test_models()
    model_params = models["smollm_135m"]  # Use SmolLM instead of GPT-2

    generation_params = GenerationParams(
        max_new_tokens=10, temperature=0.0, seed=42, batch_size=test_spec.batch_size
    )

    input = [FIXED_PROMPT] * (test_spec.num_batches * test_spec.batch_size)
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
    )

    # Validate that we got the expected number of conversations with responses
    assert len(output) == test_spec.num_batches * test_spec.batch_size
    assert validate_generation_output(output)

    # Enhanced property-based validation for non-interactive inference
    assert_response_properties(
        output,
        min_length=3,
        max_length=200,  # Short responses for fixed prompt
        expected_keywords=None,  # Don't enforce specific keywords for "Hello world!"
        forbidden_patterns=[r"\berror\b", r"\bfailed\b", r"\bunable\b"],
    )

    # Validate response relevance to the fixed prompt (use broader topic matching)
    assert_response_relevance(output)

    # Check that each conversation has the original prompt plus a response
    for conversation in output:
        assert len(conversation.messages) >= 2  # User message + Assistant response
        assert conversation.messages[0].content == FIXED_PROMPT
        assert conversation.messages[-1].role == Role.ASSISTANT
        last_msg_content = conversation.messages[-1].compute_flattened_text_content()
        assert len(last_msg_content.strip()) > 0


@pytest.mark.parametrize(
    "test_spec",
    [
        InferTestSpec(num_batches=1, batch_size=1),
        InferTestSpec(num_batches=1, batch_size=2),
    ],
    ids=_get_infer_test_spec_id,
)
@requires_cuda_initialized()
@requires_gpus()
def test_infer_basic_non_interactive_with_images(
    test_spec: InferTestSpec, root_testdata_dir: Path
):
    model_params = ModelParams(
        model_name="HuggingFaceTB/SmolVLM-Instruct",
        model_max_length=1024,
        trust_remote_code=True,
        torch_dtype_str="bfloat16",
        device_map=get_default_device_map_for_inference(),
    )
    generation_params = GenerationParams(
        max_new_tokens=10, temperature=0.0, seed=42, batch_size=test_spec.batch_size
    )

    png_image_bytes = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "the_great_wave_off_kanagawa.jpg"
    )

    test_prompt: str = "Generate a short, descriptive caption for this image!"

    input = [test_prompt] * (test_spec.num_batches * test_spec.batch_size)
    output = infer(
        config=InferenceConfig(model=model_params, generation=generation_params),
        inputs=input,
        input_image_bytes=[png_image_bytes],
    )

    # Updated for SmolVLM-Instruct - more capable 2B model
    valid_responses = [
        "A large wave",
        "The image shows",
        "This is a Japanese",
        "A Japanese art",
        "An ocean wave",
        "A famous artwork",
        "The Great Wave",
    ]

    def _create_conversation(response: str) -> Conversation:
        return Conversation(
            messages=(
                [
                    Message(
                        role=Role.USER,
                        content=[
                            ContentItem(binary=png_image_bytes, type=Type.IMAGE_BINARY),
                            ContentItem(
                                content=test_prompt,
                                type=Type.TEXT,
                            ),
                        ],
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=response,
                    ),
                ]
            )
        )

    # Check that each output conversation matches one of the valid responses
    assert len(output) == test_spec.num_batches * test_spec.batch_size
    for conv in output:
        assert any(
            _compare_conversation_lists([conv], [_create_conversation(response)])
            for response in valid_responses
        ), f"Generated response '{conv.messages[-1].content}' not in valid responses"


# Check engine availability for new tests
try:
    __import__("vllm")
    vllm_available = True
except ImportError:
    vllm_available = False

try:
    __import__("llama_cpp")
    llamacpp_available = True
except ImportError:
    llamacpp_available = False


@pytest.mark.parametrize(
    "engine_type",
    [
        InferenceEngineType.NATIVE,
        pytest.param(
            InferenceEngineType.VLLM,
            marks=pytest.mark.skipif(not vllm_available, reason="vLLM not available"),
        ),
        pytest.param(
            InferenceEngineType.LLAMACPP,
            marks=pytest.mark.skipif(
                not llamacpp_available, reason="LlamaCpp not available"
            ),
        ),
    ],
)
def test_infer_with_different_engines(engine_type: InferenceEngineType):
    """Test inference with different engines."""
    models = get_test_models()

    if engine_type == InferenceEngineType.LLAMACPP:
        # Use GGUF model for LlamaCpp
        model_params = models["gemma_270m_gguf"]
        # Reduce requirements for CPU inference
        generation_params = GenerationParams(max_new_tokens=8, temperature=0.0, seed=42)
    else:
        # Use standard model for Native and VLLM
        model_params = models["smollm_135m"]
        generation_params = GenerationParams(
            max_new_tokens=10, temperature=0.0, seed=42
        )

        if engine_type == InferenceEngineType.VLLM:
            # Skip if insufficient GPU memory for VLLM
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                available_gb = total_memory / (1024**3)
                if available_gb < 4.0:
                    pytest.skip("Insufficient VRAM for VLLM test")
            else:
                pytest.skip("CUDA not available for VLLM test")

    # Configure inference with specific engine
    config = InferenceConfig(
        model=model_params,
        generation=generation_params,
        engine=engine_type,
    )

    # Test with single input
    inputs = ["Tell me about the sky."]

    start_time = time.time()
    output = infer(config=config, inputs=inputs)
    elapsed_time = time.time() - start_time

    # Validate output
    assert len(output) == 1
    assert validate_generation_output(output)

    # Enhanced property-based validation for different engines
    assert_response_properties(
        output,
        min_length=3,
        max_length=400,
        # Make keywords optional since models may respond differently
        # to "Tell me about the sky"
        expected_keywords=None,  # Don't enforce specific keywords
        forbidden_patterns=[r"\berror\b", r"\bfailed\b", r"\bunable\b"],
    )

    # Should address the topic appropriately
    assert_response_relevance(output, expected_topics=["sky", "weather", "atmosphere"])

    # Performance validation (timeouts vary by engine)
    tokens_generated = count_response_tokens(output)
    max_time = (
        60.0 if engine_type == InferenceEngineType.LLAMACPP else 30.0
    )  # CPU vs GPU
    min_throughput = 0.5 if engine_type == InferenceEngineType.LLAMACPP else 2.0

    assert_performance_requirements(
        elapsed_time,
        tokens_generated,
        max_time_seconds=max_time,
        min_throughput=min_throughput,
    )

    # Check response content
    assert output[0].messages[0].content == inputs[0]


@pytest.mark.skipif(not vllm_available, reason="vLLM not available")
@requires_cuda_initialized()
@pytest.mark.single_gpu
def test_infer_vllm_specific_features():
    """Test VLLM-specific configuration in infer function."""
    models = get_test_models()
    model_params = models["smollm_135m"]

    # Add VLLM-specific model kwargs
    model_params.model_kwargs = {"gpu_memory_utilization": 0.6, "max_num_seqs": 8}

    config = InferenceConfig(
        model=model_params,
        generation=GenerationParams(max_new_tokens=12, temperature=0.0, seed=42),
        engine=InferenceEngineType.VLLM,
    )

    inputs = ["What is machine learning?", "Explain neural networks."]

    start_time = time.time()
    output = infer(config=config, inputs=inputs)
    elapsed_time = time.time() - start_time

    # Validate output
    assert len(output) == 2
    assert validate_generation_output(output)

    # Enhanced validation for VLLM-specific features test
    assert_response_properties(
        output,
        min_length=5,
        max_length=600,  # Longer for technical explanations
        expected_keywords=["machine", "learning", "neural", "network"],
        forbidden_patterns=[r"\berror\b", r"\bfailed\b", r"\bunable\b"],
    )

    # Should address technical topics appropriately
    assert_response_relevance(
        output, expected_topics=["machine learning", "neural networks", "technology"]
    )

    # Performance validation for VLLM features
    tokens_generated = count_response_tokens(output)
    assert_performance_requirements(
        elapsed_time,
        tokens_generated,
        max_time_seconds=35.0,
        min_throughput=3.0,  # Should be efficient with VLLM optimizations
    )

    for i, conversation in enumerate(output):
        assert conversation.messages[0].content == inputs[i]


@pytest.mark.skipif(not llamacpp_available, reason="LlamaCpp not available")
def test_infer_llamacpp_memory_optimization():
    """Test LlamaCpp with memory optimization features."""
    models = get_test_models()
    model_params = models["gemma_270m_gguf"]

    # Add LlamaCpp-specific memory optimization
    model_params.model_kwargs = {
        **model_params.model_kwargs,
        "use_mmap": True,
        "use_mlock": True,
        "n_threads": 2,
        "verbose": False,
    }

    config = InferenceConfig(
        model=model_params,
        generation=GenerationParams(max_new_tokens=8, temperature=0.0, seed=42),
        engine=InferenceEngineType.LLAMACPP,
    )

    inputs = ["Hello, how are you?"]
    output = infer(config=config, inputs=inputs)

    # Validate output
    assert len(output) == 1
    assert validate_generation_output(output)

    response = output[0].messages[-1].compute_flattened_text_content()
    assert len(response.strip()) > 1
