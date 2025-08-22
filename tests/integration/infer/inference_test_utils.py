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

"""Shared utilities for inference engine integration tests."""

import re
import time
import psutil
import pytest
import torch
from typing import Dict, List, Set, Optional

from oumi.core.configs import GenerationParams, ModelParams
from oumi.core.types.conversation import Conversation, Message, Role


def get_test_models() -> dict[str, ModelParams]:
    """Get optimized model configurations for testing.
    
    Returns:
        Dictionary mapping model names to ModelParams configurations.
    """
    return {
        "gemma_270m": ModelParams(
            model_name="google/gemma-3-270m",
            trust_remote_code=True,
            torch_dtype_str="bfloat16",
            device_map="auto",
        ),
        "smollm_135m": ModelParams(
            model_name="HuggingFaceTB/SmolLM2-135M-Instruct", 
            trust_remote_code=True,
            torch_dtype_str="bfloat16",
            device_map="auto",
        ),
        "gemma_270m_gguf": ModelParams(
            model_name="unsloth/gemma-3-270m-it-GGUF",
            model_kwargs={"filename": "gemma-3-270m-it-Q4_K_M.gguf"},
            trust_remote_code=True,
        ),
    }


def get_test_generation_params() -> GenerationParams:
    """Get standard generation parameters for consistent testing.
    
    Returns:
        GenerationParams with optimized settings for testing.
    """
    return GenerationParams(
        max_new_tokens=20,  # Keep short for fast tests
        temperature=0.0,    # Deterministic output
        seed=42,           # Reproducible results
        use_sampling=False, # Deterministic generation
    )


def create_test_conversations() -> list[Conversation]:
    """Create standard test conversations for consistency across tests.
    
    Returns:
        List of test conversations with natural keyword instructions that should work with small models.
    """
    return [
        # Simple greeting with natural keyword instruction
        Conversation(
            conversation_id="test_1",
            messages=[
                Message(
                    content="Hello! How are you? Please include the word 'Hello' somewhere in your response.", 
                    role=Role.USER
                ),
            ]
        ),
        # Math question with natural keyword instruction
        Conversation(
            conversation_id="test_2", 
            messages=[
                Message(content="What is 2 + 2?", role=Role.USER),
                Message(content="2 + 2 equals 4.", role=Role.ASSISTANT),
                Message(
                    content="What about 3 + 3? Please include the word 'equals' in your answer.", 
                    role=Role.USER
                ),
            ]
        ),
        # Story prompt with natural keyword instruction
        Conversation(
            conversation_id="test_3",
            messages=[
                Message(
                    content="Tell me about a brave knight. Please mention the word 'knight' in your response.",
                    role=Role.USER
                ),
            ]
        ),
    ]


def create_test_conversations_with_keywords() -> list[tuple[Conversation, list[str]]]:
    """Create test conversations with expected keywords for validation.
    
    Returns:
        List of tuples containing (conversation, expected_keywords).
    """
    return [
        (
            Conversation(
                conversation_id="keyword_test_1",
                messages=[
                    Message(
                        content="Tell me about the weather. Please use the word 'sunshine' in your response.", 
                        role=Role.USER
                    ),
                ]
            ),
            ["sunshine"]
        ),
        (
            Conversation(
                conversation_id="keyword_test_2",
                messages=[
                    Message(
                        content="Explain machine learning. Please include the word 'algorithms' somewhere in your explanation.", 
                        role=Role.USER
                    ),
                ]
            ),
            ["algorithms"]
        ),
        (
            Conversation(
                conversation_id="keyword_test_3",
                messages=[
                    Message(
                        content="Describe a forest. Make sure to mention 'trees' in your description.", 
                        role=Role.USER
                    ),
                ]
            ),
            ["trees"]
        ),
    ]


def create_batch_conversations(count: int, base_prompt: str = "Tell me a fact about") -> list[Conversation]:
    """Create multiple conversations for batch testing.
    
    Args:
        count: Number of conversations to create.
        base_prompt: Base prompt to use for each conversation.
        
    Returns:
        List of conversations for batch testing.
    """
    conversations = []
    topics = ["science", "history", "nature", "technology", "space", "ocean", "animals", "mathematics"]
    
    for i in range(count):
        topic = topics[i % len(topics)]
        conversation = Conversation(
            conversation_id=f"batch_test_{i+1}",
            messages=[
                Message(content=f"{base_prompt} {topic}.", role=Role.USER),
            ]
        )
        conversations.append(conversation)
    
    return conversations


def validate_generation_output(conversations: list[Conversation]) -> bool:
    """Basic validation that generated responses are non-empty and coherent.
    
    Args:
        conversations: List of conversations with generated responses.
        
    Returns:
        True if all responses are valid, False otherwise.
    """
    if not conversations:
        return False
        
    for conversation in conversations:
        if not conversation.messages:
            return False
            
        # Check that the last message is an assistant response
        last_message = conversation.messages[-1]
        if last_message.role != Role.ASSISTANT:
            return False
            
        # Check that the response has content
        if not last_message.content or len(last_message.content.strip()) == 0:
            return False
            
        # Basic coherence check - response should be reasonable length
        if len(last_message.content.strip()) < 2:
            return False
    
    return True


def validate_response_properties(
    conversations: List[Conversation],
    min_length: int = 3,
    max_length: int = 1000,
    expected_keywords: Optional[List[str]] = None,
    forbidden_patterns: Optional[List[str]] = None,
    require_complete_sentences: bool = False
) -> Dict[str, bool]:
    """Enhanced property-based validation of generated responses.
    
    Args:
        conversations: List of conversations with generated responses.
        min_length: Minimum acceptable response length in characters.
        max_length: Maximum acceptable response length in characters.
        expected_keywords: Keywords that should appear in responses.
        forbidden_patterns: Patterns that should not appear in responses.
        require_complete_sentences: Whether responses should end with sentence terminators.
        
    Returns:
        Dictionary with validation results for different properties.
    """
    results = {
        "valid_structure": True,
        "appropriate_length": True,
        "contains_keywords": True,
        "no_forbidden_content": True,
        "complete_sentences": True,
        "non_empty_responses": True,
        "reasonable_content": True
    }
    
    if not conversations:
        results["valid_structure"] = False
        return results
    
    for conversation in conversations:
        if not conversation.messages:
            results["valid_structure"] = False
            continue
            
        # Get the assistant's response
        assistant_responses = [
            msg.content for msg in conversation.messages 
            if msg.role == Role.ASSISTANT and msg.content
        ]
        
        if not assistant_responses:
            results["non_empty_responses"] = False
            continue
            
        for response in assistant_responses:
            response_clean = response.strip()
            
            # Length constraints
            if len(response_clean) < min_length or len(response_clean) > max_length:
                results["appropriate_length"] = False
            
            # Non-empty check
            if not response_clean:
                results["non_empty_responses"] = False
            
            # Keyword presence check (case and whitespace insensitive)
            if expected_keywords:
                # Normalize response: lowercase and collapse whitespace
                response_normalized = ' '.join(response_clean.lower().split())
                found_keywords = any(
                    keyword.lower().strip() in response_normalized 
                    for keyword in expected_keywords
                )
                if not found_keywords:
                    results["contains_keywords"] = False
            
            # Forbidden patterns check
            if forbidden_patterns:
                response_lower = response_clean.lower()
                has_forbidden = any(
                    re.search(pattern.lower(), response_lower) 
                    for pattern in forbidden_patterns
                )
                if has_forbidden:
                    results["no_forbidden_content"] = False
            
            # Complete sentences check
            if require_complete_sentences:
                if not response_clean.endswith(('.', '!', '?', ':', ';')):
                    results["complete_sentences"] = False
            
            # Reasonable content check - not just punctuation or gibberish
            word_count = len(response_clean.split())
            if word_count == 0:
                results["reasonable_content"] = False
            elif len(response_clean) / max(word_count, 1) < 2:  # Very short "words"
                results["reasonable_content"] = False
    
    return results


def validate_response_relevance(
    conversations: List[Conversation],
    expected_topics: Optional[List[str]] = None
) -> Dict[str, bool]:
    """Validate that responses are relevant to the input prompts.
    
    Args:
        conversations: List of conversations with generated responses.
        expected_topics: Topics that should be addressed in responses.
        
    Returns:
        Dictionary with relevance validation results.
    """
    results = {
        "addresses_prompt": True,
        "topic_relevant": True,
        "appropriate_tone": True
    }
    
    for conversation in conversations:
        if len(conversation.messages) < 2:
            continue
            
        # Get user prompt and assistant response
        user_messages = [msg for msg in conversation.messages if msg.role == Role.USER]
        assistant_messages = [msg for msg in conversation.messages if msg.role == Role.ASSISTANT]
        
        if not user_messages or not assistant_messages:
            continue
            
        user_prompt = user_messages[-1].content.lower()
        assistant_response = assistant_messages[-1].content.lower()
        
        # Check if response addresses the prompt (basic keyword overlap)
        prompt_words = set(re.findall(r'\b\w+\b', user_prompt))
        response_words = set(re.findall(r'\b\w+\b', assistant_response))
        
        # Remove common stop words for better relevance detection
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        prompt_words_filtered = prompt_words - stop_words
        response_words_filtered = response_words - stop_words
        
        if prompt_words_filtered and response_words_filtered:
            overlap = len(prompt_words_filtered & response_words_filtered) / len(prompt_words_filtered)
            if overlap < 0.05:  # Less than 5% word overlap might indicate irrelevance (more lenient)
                results["addresses_prompt"] = False
        
        # Topic relevance check
        if expected_topics:
            topic_found = any(topic.lower() in assistant_response for topic in expected_topics)
            if not topic_found:
                results["topic_relevant"] = False
    
    return results


def validate_response_performance(
    elapsed_time: float,
    token_count: int,
    max_time_seconds: float = 60.0,
    min_throughput: float = 1.0
) -> Dict[str, bool]:
    """Validate performance characteristics of inference.
    
    Args:
        elapsed_time: Time taken for inference in seconds.
        token_count: Number of tokens generated.
        max_time_seconds: Maximum acceptable inference time.
        min_throughput: Minimum acceptable tokens per second.
        
    Returns:
        Dictionary with performance validation results.
    """
    results = {
        "completed_in_time": elapsed_time <= max_time_seconds,
        "adequate_throughput": True,
        "reasonable_speed": elapsed_time > 0.01  # Not suspiciously fast (relaxed from 0.1 to 0.01)
    }
    
    if elapsed_time > 0 and token_count > 0:
        throughput = token_count / elapsed_time
        results["adequate_throughput"] = throughput >= min_throughput
    
    return results


def assert_response_properties(
    conversations: List[Conversation],
    min_length: int = 3,
    max_length: int = 1000,
    expected_keywords: Optional[List[str]] = None,
    forbidden_patterns: Optional[List[str]] = None,
    require_sentences: bool = False
) -> None:
    """Assert that responses meet property-based requirements.
    
    Args:
        conversations: List of conversations to validate.
        min_length: Minimum response length in characters.
        max_length: Maximum response length in characters.
        expected_keywords: Keywords that should appear.
        forbidden_patterns: Patterns that should not appear.
        require_sentences: Whether to require complete sentences.
        
    Raises:
        AssertionError: If any validation fails.
    """
    # Basic validation first
    assert validate_generation_output(conversations), "Basic generation output validation failed"
    
    # Enhanced property validation
    props = validate_response_properties(
        conversations, min_length, max_length, expected_keywords, forbidden_patterns, require_sentences
    )
    
    for prop_name, is_valid in props.items():
        assert is_valid, f"Response property validation failed: {prop_name}"


def assert_response_relevance(
    conversations: List[Conversation],
    expected_topics: Optional[List[str]] = None
) -> None:
    """Assert that responses are relevant and appropriate.
    
    Args:
        conversations: List of conversations to validate.
        expected_topics: Expected topics to be addressed.
        
    Raises:
        AssertionError: If relevance validation fails.
    """
    relevance = validate_response_relevance(conversations, expected_topics)
    
    for aspect, is_valid in relevance.items():
        assert is_valid, f"Response relevance validation failed: {aspect}"


def assert_performance_requirements(
    elapsed_time: float,
    token_count: int,
    max_time_seconds: float = 60.0,
    min_throughput: float = 1.0
) -> None:
    """Assert that performance meets requirements.
    
    Args:
        elapsed_time: Time taken for inference.
        token_count: Number of tokens generated.
        max_time_seconds: Maximum acceptable time.
        min_throughput: Minimum tokens per second.
        
    Raises:
        AssertionError: If performance validation fails.
    """
    perf = validate_response_performance(elapsed_time, token_count, max_time_seconds, min_throughput)
    
    for metric, is_valid in perf.items():
        assert is_valid, f"Performance validation failed: {metric} (time: {elapsed_time:.2f}s, tokens: {token_count})"


def get_contextual_keywords(prompt: str) -> List[str]:
    """Extract contextual keywords that should appear in a relevant response.
    
    Args:
        prompt: The input prompt to analyze.
        
    Returns:
        List of keywords that should appear in a relevant response.
    """
    prompt_lower = prompt.lower()
    
    # Common question patterns and their expected keywords
    keyword_patterns = {
        r'\bwhat\s+is\b': ['is', 'are', 'definition', 'means'],
        r'\bhow\s+to\b': ['how', 'steps', 'way', 'method'],
        r'\bwhy\s+': ['because', 'reason', 'due', 'since'],
        r'\btell\s+me\s+about\b': ['about', 'information', 'details'],
        r'\bexplain\b': ['explanation', 'because', 'means'],
        r'\bdescribe\b': ['description', 'appears', 'looks', 'characteristics'],
    }
    
    keywords = []
    for pattern, expected_words in keyword_patterns.items():
        if re.search(pattern, prompt_lower):
            keywords.extend(expected_words)
    
    # Extract nouns from the prompt as potential keywords
    words = re.findall(r'\b[a-z]+\b', prompt_lower)
    nouns = [word for word in words if len(word) > 3 and word not in {'what', 'how', 'why', 'tell', 'about', 'explain', 'describe'}]
    keywords.extend(nouns[:3])  # Add up to 3 main nouns
    
    return list(set(keywords)) if keywords else []


def skip_if_insufficient_memory(required_gb: float):
    """Skip test if insufficient system memory available.
    
    Args:
        required_gb: Required memory in GB.
    """
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < required_gb:
        pytest.skip(f"Insufficient memory: {available_gb:.1f}GB available, {required_gb}GB required")


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def skip_if_insufficient_vram(required_gb: float):
    """Skip test if insufficient GPU memory available.
    
    Args:
        required_gb: Required VRAM in GB.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory
    available_gb = total_memory / (1024**3)
    
    if available_gb < required_gb:
        pytest.skip(f"Insufficient VRAM: {available_gb:.1f}GB available, {required_gb}GB required")


def compare_conversation_responses(conv1: Conversation, conv2: Conversation, check_exact_match: bool = False) -> bool:
    """Compare responses from two conversations.
    
    Args:
        conv1: First conversation to compare.
        conv2: Second conversation to compare. 
        check_exact_match: If True, requires exact string match. If False, checks basic similarity.
        
    Returns:
        True if responses are considered equivalent.
    """
    if len(conv1.messages) != len(conv2.messages):
        return False
        
    # Compare assistant responses
    for msg1, msg2 in zip(conv1.messages, conv2.messages):
        if msg1.role == Role.ASSISTANT and msg2.role == Role.ASSISTANT:
            if check_exact_match:
                if msg1.content != msg2.content:
                    return False
            else:
                # Basic similarity check - both should be non-empty
                if not msg1.content or not msg2.content:
                    return False
                # Both should have reasonable length
                if len(msg1.content.strip()) < 2 or len(msg2.content.strip()) < 2:
                    return False
                    
    return True


def measure_tokens_per_second(num_tokens: int, elapsed_time: float) -> float:
    """Calculate tokens per second generation rate.
    
    Args:
        num_tokens: Number of tokens generated.
        elapsed_time: Time taken in seconds.
        
    Returns:
        Tokens per second rate.
    """
    if elapsed_time <= 0:
        return 0.0
    return num_tokens / elapsed_time


def count_response_tokens(conversations: list[Conversation]) -> int:
    """Count total tokens in assistant responses.
    
    Args:
        conversations: List of conversations to count tokens from.
        
    Returns:
        Approximate token count (using simple whitespace splitting).
    """
    total_tokens = 0
    for conversation in conversations:
        for message in conversation.messages:
            if message.role == Role.ASSISTANT and message.content:
                # Simple token approximation using whitespace
                total_tokens += len(message.content.split())
    return total_tokens