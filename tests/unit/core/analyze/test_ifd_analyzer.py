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

"""Tests for the IFDAnalyzer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch


# Check if transformers is available
def _transformers_available():
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


# Skip all tests if transformers is not available
pytestmark = pytest.mark.skipif(
    not _transformers_available(),
    reason="transformers not installed",
)


class TestIFDAnalyzerInit:
    """Tests for IFDAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()
        assert analyzer.model_name == "Qwen/Qwen3-0.6B"
        assert analyzer.batch_size == 4
        assert analyzer.max_length == 2048
        assert analyzer.trust_remote_code is True
        assert analyzer.low_memory is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(
            model_name="gpt2",
            batch_size=8,
            max_length=1024,
            instruction_column="prompt",
            response_column="completion",
            trust_remote_code=False,
            low_memory=True,
        )
        assert analyzer.model_name == "gpt2"
        assert analyzer.batch_size == 8
        assert analyzer.max_length == 1024
        assert analyzer.instruction_column == "prompt"
        assert analyzer.response_column == "completion"
        assert analyzer.trust_remote_code is False
        assert analyzer.low_memory is True

    def test_init_device_auto_detection(self):
        """Test that device is auto-detected when not specified."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()
        # Device should be set based on availability
        assert analyzer.device in ["cuda", "mps", "cpu"]

    def test_init_explicit_device(self):
        """Test initialization with explicit device."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(device="cpu")
        assert analyzer.device == "cpu"

    def test_init_valid_torch_dtype(self):
        """Test initialization with valid torch dtype."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(torch_dtype="float32")
        assert analyzer._torch_dtype == torch.float32

        analyzer = IFDAnalyzer(torch_dtype="float16")
        assert analyzer._torch_dtype == torch.float16

        analyzer = IFDAnalyzer(torch_dtype="bfloat16")
        assert analyzer._torch_dtype == torch.bfloat16

    def test_init_invalid_torch_dtype(self):
        """Test that invalid torch dtype raises error."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        with pytest.raises(ValueError, match="Invalid torch_dtype"):
            IFDAnalyzer(torch_dtype="invalid")


class TestIFDAnalyzerColumnDetection:
    """Tests for instruction/response column detection."""

    def test_find_columns_explicit(self):
        """Test finding columns when explicitly specified."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(
            instruction_column="my_prompt",
            response_column="my_response",
        )

        df = pd.DataFrame({
            "my_prompt": ["instruction"],
            "my_response": ["response"],
        })

        inst_col, resp_col = analyzer._find_instruction_response_columns(df)
        assert inst_col == "my_prompt"
        assert resp_col == "my_response"

    def test_find_columns_auto_detect_instruction(self):
        """Test auto-detection of instruction column."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()

        # Test 'instruction' pattern
        df = pd.DataFrame({
            "instruction": ["inst"],
            "output": ["out"],
        })
        inst_col, resp_col = analyzer._find_instruction_response_columns(df)
        assert inst_col == "instruction"
        assert resp_col == "output"

    def test_find_columns_auto_detect_prompt(self):
        """Test auto-detection of prompt column."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()

        df = pd.DataFrame({
            "prompt": ["prompt text"],
            "response": ["response text"],
        })
        inst_col, resp_col = analyzer._find_instruction_response_columns(df)
        assert inst_col == "prompt"
        assert resp_col == "response"

    def test_find_columns_returns_none_when_not_found(self):
        """Test that None is returned when columns can't be found."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()

        df = pd.DataFrame({
            "col1": ["data"],
            "col2": ["data"],
        })
        inst_col, resp_col = analyzer._find_instruction_response_columns(df)
        assert inst_col is None
        assert resp_col is None


class TestIFDAnalyzerWithMocks:
    """Tests using mocked model and tokenizer."""

    def _create_mock_model(self, loss_value: float = 2.0):
        """Create a mock causal LM model."""
        model = MagicMock()
        # Mock the forward pass to return a loss
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = loss_value
        model.return_value = mock_output
        model.to = MagicMock(return_value=model)
        model.eval = MagicMock(return_value=model)
        model.parameters = MagicMock(return_value=iter([torch.tensor([1.0])]))
        return model

    def _create_mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        # Mock encoding
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        tokenizer.return_value = mock_encoding

        # For prefix tokenization
        tokenizer.pad_token = "[PAD]"
        tokenizer.pad_token_id = 0
        tokenizer.eos_token = "[EOS]"
        tokenizer.eos_token_id = 1

        return tokenizer

    def test_compute_perplexity_without_prefix(self):
        """Test perplexity computation without prefix."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(device="cpu")
        # Directly inject mocks to bypass _load_model
        analyzer._model = self._create_mock_model(loss_value=2.0)
        analyzer._tokenizer = self._create_mock_tokenizer()

        result = analyzer._compute_perplexity("test response")

        assert "perplexity" in result
        assert "loss" in result
        assert result["loss"] == 2.0
        # exp(2.0) ≈ 7.389
        assert abs(result["perplexity"] - 7.389) < 0.01

    def test_compute_perplexity_with_prefix(self):
        """Test perplexity computation with prefix (instruction)."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(device="cpu")
        analyzer._model = self._create_mock_model(loss_value=1.5)

        # Set up mock tokenizer to return different lengths for full text vs prefix
        mock_tokenizer = MagicMock()

        def mock_tokenize(text, **kwargs):
            mock_encoding = MagicMock()
            if "test instruction" in text:
                # Full text (instruction + response)
                mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
                mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
            else:
                # Prefix only or response only
                mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4]])
                mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1]])
            return mock_encoding

        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.eos_token_id = 1

        analyzer._tokenizer = mock_tokenizer

        result = analyzer._compute_perplexity(
            "test response", prefix="test instruction"
        )

        assert "perplexity" in result
        assert "loss" in result
        assert result["loss"] == 1.5

    def test_compute_ifd_for_sample(self):
        """Test IFD computation for a single sample."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(device="cpu")

        # Set up mock model to return different losses for with/without instruction
        call_count = [0]

        def mock_forward(**kwargs):
            call_count[0] += 1
            mock_output = MagicMock()
            if call_count[0] % 2 == 1:  # First call (with instruction)
                mock_output.loss = MagicMock()
                mock_output.loss.item.return_value = 1.5  # Lower loss with instruction
            else:  # Second call (without instruction)
                mock_output.loss = MagicMock()
                mock_output.loss.item.return_value = 3.0  # Higher loss without
            return mock_output

        mock_model = MagicMock()
        mock_model.side_effect = mock_forward

        analyzer._model = mock_model
        analyzer._tokenizer = self._create_mock_tokenizer()

        result = analyzer._compute_ifd_for_sample(
            instruction="Write a poem",
            response="Roses are red, violets are blue",
        )

        assert "ifd_score" in result
        assert "ppl_with_instruction" in result
        assert "ppl_without_instruction" in result
        assert "response_loss" in result

        # IFD should be > 1 since instruction helps
        # IFD = PPL(without) / PPL(with) = exp(3.0) / exp(1.5)
        # ≈ 20.09 / 4.48 ≈ 4.48
        assert result["ifd_score"] > 1.0

    def test_analyze_sample_with_valid_columns(self):
        """Test full analysis with valid DataFrame columns."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(device="cpu")
        analyzer.analyzer_id = "ifd"

        # Inject mocks
        analyzer._model = self._create_mock_model(loss_value=2.0)
        analyzer._tokenizer = self._create_mock_tokenizer()

        df = pd.DataFrame({
            "instruction": ["Write a poem", "Explain gravity"],
            "output": ["Roses are red...", "Gravity is a force..."],
        })

        result_df = analyzer.analyze_sample(df)

        # Check that IFD columns are added
        assert "ifd_score" in result_df.columns
        assert "ifd_ppl_with_instruction" in result_df.columns
        assert "ifd_ppl_without_instruction" in result_df.columns
        assert "ifd_response_loss" in result_df.columns

        # Check that values are computed for all rows
        assert len(result_df) == 2


class TestIFDAnalyzerConversationFormat:
    """Tests for conversation format handling."""

    def test_is_conversation_format_true(self):
        """Test detection of conversation format."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()

        df = pd.DataFrame({
            "text_content": ["Hello", "Hi there"],
            "role": ["user", "assistant"],
            "conversation_index": [0, 0],
        })

        assert analyzer._is_conversation_format(df) is True

    def test_is_conversation_format_false(self):
        """Test detection when not conversation format."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()

        df = pd.DataFrame({
            "instruction": ["Write a poem"],
            "output": ["Roses are red..."],
        })

        assert analyzer._is_conversation_format(df) is False

    def test_analyze_conversation_format(self):
        """Test analysis of conversation format DataFrame."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer(device="cpu")
        analyzer.analyzer_id = "ifd"

        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.loss = MagicMock()
        mock_output.loss.item.return_value = 2.0
        mock_model.return_value = mock_output

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        mock_encoding.attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
        mock_tokenizer.return_value = mock_encoding
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.pad_token_id = 0

        analyzer._model = mock_model
        analyzer._tokenizer = mock_tokenizer

        # Create conversation format DataFrame
        df = pd.DataFrame({
            "text_content": ["What is Python?", "Python is a programming language."],
            "role": ["user", "assistant"],
            "conversation_index": [0, 0],
            "message_index": [0, 1],
        })

        result_df = analyzer.analyze_sample(df)

        # IFD columns should be added
        assert "ifd_score" in result_df.columns

        # Only assistant messages should have IFD scores
        assert result_df.loc[0, "ifd_score"] is None  # user message
        assert result_df.loc[1, "ifd_score"] is not None  # assistant message


class TestIFDAnalyzerValidation:
    """Tests for input validation."""

    def test_missing_columns_logs_warning(self):
        """Test that missing columns logs a warning and returns unchanged DataFrame."""
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer

        analyzer = IFDAnalyzer()

        df = pd.DataFrame({
            "col1": ["data"],
            "col2": ["data"],
        })

        with patch("oumi.core.analyze.ifd_analyzer.logger") as mock_logger:
            result_df = analyzer.analyze_sample(df)

            mock_logger.warning.assert_called()
            # DataFrame should be unchanged (only original columns)
            assert "col1" in result_df.columns
            assert "col2" in result_df.columns


class TestIFDAnalyzerInterpretation:
    """Tests for IFD score interpretation."""

    def test_high_ifd_indicates_valuable_sample(self):
        """Test that high IFD score indicates a valuable sample."""
        # High IFD means: PPL(without instruction) >> PPL(with instruction)
        # This means the instruction provides significant guidance
        ppl_without = 100.0  # High perplexity without instruction
        ppl_with = 10.0  # Low perplexity with instruction
        ifd = ppl_without / ppl_with

        assert ifd == 10.0
        assert ifd > 1.0  # Valuable sample

    def test_low_ifd_indicates_potentially_problematic(self):
        """Test that low IFD (near or below 1) may indicate issues."""
        # Low IFD means instruction doesn't help much
        ppl_without = 10.0
        ppl_with = 10.0  # Same perplexity
        ifd = ppl_without / ppl_with

        assert ifd == 1.0  # Instruction provides no guidance

        # IFD < 1 means instruction makes prediction worse (problematic)
        ppl_without = 5.0
        ppl_with = 10.0  # Higher perplexity WITH instruction
        ifd = ppl_without / ppl_with

        assert ifd == 0.5
        assert ifd < 1.0  # Problematic sample


class TestIFDAnalyzerRegistry:
    """Tests for analyzer registry integration."""

    def test_analyzer_is_registered(self):
        """Test that IFDAnalyzer is registered with the correct ID."""
        from oumi.core.registry import REGISTRY

        # Import to trigger registration
        from oumi.core.analyze.ifd_analyzer import IFDAnalyzer  # noqa: F401

        # Check registration
        registered_analyzer = REGISTRY.get_sample_analyzer("ifd")
        assert registered_analyzer is not None
        assert registered_analyzer is IFDAnalyzer
