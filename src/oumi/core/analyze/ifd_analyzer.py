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

"""IFD (Instruction-Following Difficulty) analyzer for data quality filtering.

This analyzer implements the IFD metric from Cherry LLM and Superfiltering papers
for assessing instruction-response pair quality.

References:
    - Cherry LLM: https://github.com/tianyi-lab/Cherry_LLM
    - Superfiltering: https://arxiv.org/html/2402.00530v1
"""

import math
from typing import Any, Optional

import pandas as pd
import torch

from oumi.core.analyze.sample_analyzer import SampleAnalyzer
from oumi.core.registry import register_sample_analyzer
from oumi.utils.logging import logger


@register_sample_analyzer("ifd")
class IFDAnalyzer(SampleAnalyzer):
    """Analyzer for computing Instruction-Following Difficulty (IFD) scores.

    IFD measures how much an instruction helps a language model predict
    the response. It's computed as:

        IFD = PPL(response | no instruction) / PPL(response | with instruction)

    Where PPL is perplexity. Higher IFD values indicate that the instruction
    provides more guidance to the model, making the sample more valuable for
    training.

    Key findings from the research:
        - 5-10% of data selected via IFD can match full-data performance
        - Weak models (e.g., Qwen3-0.6b) can effectively filter for stronger models
        - Samples with IFD < 1.0 may have issues (instruction doesn't help)

    Output metrics:
        - ifd_score: The IFD ratio (higher = more valuable)
        - ppl_with_instruction: Perplexity with instruction context
        - ppl_without_instruction: Perplexity without instruction (response only)
        - response_loss: Cross-entropy loss on response tokens

    Example:
        >>> analyzer = IFDAnalyzer(model_name="Qwen/Qwen3-0.6B")
        >>> result_df = analyzer.analyze_sample(df, schema)
    """

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 2048,
        instruction_column: Optional[str] = None,
        response_column: Optional[str] = None,
        trust_remote_code: bool = True,
        low_memory: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize the IFDAnalyzer.

        Args:
            model_name: HuggingFace model name for perplexity calculation.
                Smaller models like Qwen3-0.6B work well and are efficient.
            device: Device to run inference on ('cuda', 'cpu', 'mps', or None for auto).
            torch_dtype: Data type for model weights ('float16', 'bfloat16', 'float32').
                If None, uses the model's default dtype.
            batch_size: Number of samples to process in each batch.
            max_length: Maximum sequence length for tokenization.
            instruction_column: Explicit column name for instructions. If None,
                auto-detects from schema (looks for 'user' role messages).
            response_column: Explicit column name for responses. If None,
                auto-detects from schema (looks for 'assistant' role messages).
            trust_remote_code: Whether to trust remote code when loading model.
            low_memory: If True, uses memory-efficient settings.
            cache_dir: Directory to cache model files.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.instruction_column = instruction_column
        self.response_column = response_column
        self.trust_remote_code = trust_remote_code
        self.low_memory = low_memory
        self.cache_dir = cache_dir

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Parse torch dtype
        self._torch_dtype = None
        if torch_dtype is not None:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if torch_dtype in dtype_map:
                self._torch_dtype = dtype_map[torch_dtype]
            else:
                raise ValueError(
                    f"Invalid torch_dtype: {torch_dtype}. "
                    f"Must be one of: {list(dtype_map.keys())}"
                )

        # Lazy-loaded model and tokenizer
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                f"transformers package is required for IFDAnalyzer: {e}. "
                "Install with: pip install transformers"
            ) from e

        logger.info(f"Loading model for IFD analysis: {self.model_name}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            cache_dir=self.cache_dir,
        )

        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Load model
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
        }

        if self._torch_dtype is not None:
            model_kwargs["torch_dtype"] = self._torch_dtype
        elif self.device == "cuda":
            # Default to float16 on GPU for efficiency
            model_kwargs["torch_dtype"] = torch.float16

        if self.low_memory:
            model_kwargs["low_cpu_mem_usage"] = True

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        self._model.to(self.device)  # type: ignore[arg-type]
        self._model.eval()
        for param in self._model.parameters():
            param.requires_grad = False

        logger.info(
            f"Loaded {self.model_name} on {self.device} "
            f"(dtype: {next(self._model.parameters()).dtype})"
        )

    def _compute_perplexity(
        self,
        text: str,
        prefix: Optional[str] = None,
    ) -> dict[str, float]:
        """Compute perplexity for text, optionally with a prefix.

        Args:
            text: The text to compute perplexity for (typically the response).
            prefix: Optional prefix text (typically the instruction).
                If provided, we compute perplexity of `text` given `prefix`.

        Returns:
            Dictionary with perplexity and loss values.
        """
        self._load_model()

        if self._tokenizer is None or self._model is None:
            return {"perplexity": float("inf"), "loss": float("inf")}

        # Prepare input text
        if prefix:
            # Tokenize prefix and text separately to identify response tokens
            full_text = prefix + text
        else:
            full_text = text

        # Tokenize
        encodings = self._tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        input_ids = encodings.input_ids.to(self.device)
        attention_mask = encodings.attention_mask.to(self.device)

        # If we have a prefix, we need to mask the prefix tokens from the loss
        if prefix:
            prefix_encodings = self._tokenizer(
                prefix,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
                add_special_tokens=False,
            )
            prefix_len = prefix_encodings.input_ids.shape[1]
        else:
            prefix_len = 0

        # Create labels (shift is handled internally by the model)
        labels = input_ids.clone()

        # Mask prefix tokens in labels (set to -100 to ignore in loss)
        if prefix_len > 0:
            # We mask up to prefix_len tokens
            labels[:, :prefix_len] = -100

        # Compute loss
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss.item()

        # Compute perplexity
        perplexity = math.exp(loss) if loss < 100 else float("inf")

        return {
            "perplexity": perplexity,
            "loss": loss,
        }

    def _compute_ifd_for_sample(
        self,
        instruction: str,
        response: str,
    ) -> dict[str, Any]:
        """Compute IFD score for an instruction-response pair.

        Args:
            instruction: The instruction/prompt text.
            response: The response/completion text.

        Returns:
            Dictionary with IFD metrics.
        """
        # Compute perplexity with instruction
        with_instruction = self._compute_perplexity(response, prefix=instruction)

        # Compute perplexity without instruction (response only)
        without_instruction = self._compute_perplexity(response, prefix=None)

        # Compute IFD score
        ppl_with = with_instruction["perplexity"]
        ppl_without = without_instruction["perplexity"]

        if ppl_with > 0 and ppl_with != float("inf"):
            ifd_score = ppl_without / ppl_with
        else:
            ifd_score = 0.0

        ppl_with_rounded = round(ppl_with, 4) if ppl_with != float("inf") else None
        ppl_without_rounded = (
            round(ppl_without, 4) if ppl_without != float("inf") else None
        )

        return {
            "ifd_score": round(ifd_score, 4),
            "ppl_with_instruction": ppl_with_rounded,
            "ppl_without_instruction": ppl_without_rounded,
            "response_loss": round(with_instruction["loss"], 4),
        }

    def _find_instruction_response_columns(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Find instruction and response columns from schema or DataFrame.

        Args:
            df: Input DataFrame.
            schema: Column schema dict.

        Returns:
            Tuple of (instruction_column, response_column) or (None, None).
        """
        # Use explicit columns if provided
        if self.instruction_column and self.response_column:
            inst_in_df = self.instruction_column in df.columns
            resp_in_df = self.response_column in df.columns
            if inst_in_df and resp_in_df:
                return self.instruction_column, self.response_column

        # Common column name patterns
        instruction_patterns = ["instruction", "prompt", "input", "question", "query"]
        response_patterns = ["response", "output", "answer", "completion"]

        instruction_col = None
        response_col = None

        # Search in columns
        for col in df.columns:
            col_lower = col.lower()
            if instruction_col is None:
                for pattern in instruction_patterns:
                    if pattern in col_lower:
                        instruction_col = col
                        break
            if response_col is None:
                for pattern in response_patterns:
                    if pattern in col_lower:
                        response_col = col
                        break

        return instruction_col, response_col

    def _is_conversation_format(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame is in conversation format (message-level).

        Conversation format has columns like: text_content, role, conversation_index
        """
        required_cols = ["text_content", "role"]
        return all(col in df.columns for col in required_cols)

    def _analyze_conversation_format(
        self,
        df: pd.DataFrame,
        analyzer_id: str,
    ) -> pd.DataFrame:
        """Analyze DataFrame in conversation format.

        For each assistant message, pairs it with the preceding user message(s)
        as the instruction context.

        Args:
            df: DataFrame with text_content, role, conversation_index columns.
            analyzer_id: ID prefix for output columns.

        Returns:
            DataFrame with IFD columns added for assistant messages.
        """
        result_df = df.copy()

        # Initialize result columns with None
        result_df[f"{analyzer_id}_score"] = None
        result_df[f"{analyzer_id}_ppl_with_instruction"] = None
        result_df[f"{analyzer_id}_ppl_without_instruction"] = None
        result_df[f"{analyzer_id}_response_loss"] = None

        # Group by conversation
        conv_col = "conversation_index" if "conversation_index" in df.columns else None

        if conv_col is None:
            logger.warning(
                "No conversation_index column found. "
                "Treating all messages as a single conversation."
            )
            conversations = [(0, df)]
        else:
            conversations = list(df.groupby(conv_col))

        total_assistant_msgs = 0
        processed = 0

        for conv_id, conv_df in conversations:
            # Sort by message index if available
            if "message_index" in conv_df.columns:
                conv_df = conv_df.sort_values("message_index")

            # Get indices of assistant messages
            assistant_mask = conv_df["role"].str.lower() == "assistant"
            assistant_indices = conv_df[assistant_mask].index.tolist()
            total_assistant_msgs += len(assistant_indices)

            for asst_idx in assistant_indices:
                # Get the assistant message (response)
                response = str(conv_df.loc[asst_idx, "text_content"])

                # Get preceding messages as instruction context
                asst_pos = conv_df.index.get_loc(asst_idx)
                preceding_df = conv_df.iloc[:asst_pos]

                # Build instruction from preceding user messages
                # (could also include system messages)
                user_messages = preceding_df[
                    preceding_df["role"].str.lower() == "user"
                ]["text_content"].tolist()

                if not user_messages:
                    # No user message before this assistant message
                    continue

                # Use the last user message as instruction
                # (or concatenate all for multi-turn context)
                instruction = str(user_messages[-1])

                try:
                    ifd_result = self._compute_ifd_for_sample(instruction, response)

                    result_df.loc[asst_idx, f"{analyzer_id}_score"] = (
                        ifd_result["ifd_score"]
                    )
                    result_df.loc[asst_idx, f"{analyzer_id}_ppl_with_instruction"] = (
                        ifd_result["ppl_with_instruction"]
                    )
                    ppl_without_col = f"{analyzer_id}_ppl_without_instruction"
                    result_df.loc[asst_idx, ppl_without_col] = (
                        ifd_result["ppl_without_instruction"]
                    )
                    result_df.loc[asst_idx, f"{analyzer_id}_response_loss"] = (
                        ifd_result["response_loss"]
                    )
                    processed += 1

                except Exception as e:
                    logger.warning(
                        f"Error computing IFD for conversation {conv_id}, "
                        f"message {asst_idx}: {e}"
                    )

            # Log progress
            if processed > 0 and processed % 100 == 0:
                logger.info(f"Processed {processed}/{total_assistant_msgs} responses")

        logger.info(
            f"IFD analysis complete. Processed {processed} assistant messages "
            f"out of {total_assistant_msgs} total."
        )

        return result_df

    def _analyze_flat_format(
        self,
        df: pd.DataFrame,
        instruction_col: str,
        response_col: str,
        analyzer_id: str,
    ) -> pd.DataFrame:
        """Analyze DataFrame in flat instruction-response format.

        Args:
            df: DataFrame with instruction and response columns.
            instruction_col: Name of instruction column.
            response_col: Name of response column.
            analyzer_id: ID prefix for output columns.

        Returns:
            DataFrame with IFD columns added.
        """
        result_df = df.copy()
        results = []

        for idx in range(len(df)):
            instruction = str(df.iloc[idx][instruction_col])
            response = str(df.iloc[idx][response_col])

            try:
                ifd_result = self._compute_ifd_for_sample(instruction, response)
            except Exception as e:
                logger.warning(f"Error computing IFD for sample {idx}: {e}")
                ifd_result = {
                    "ifd_score": None,
                    "ppl_with_instruction": None,
                    "ppl_without_instruction": None,
                    "response_loss": None,
                }

            results.append(ifd_result)

            # Log progress
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} samples")

        # Add columns to result DataFrame
        result_df[f"{analyzer_id}_score"] = [r["ifd_score"] for r in results]
        result_df[f"{analyzer_id}_ppl_with_instruction"] = [
            r["ppl_with_instruction"] for r in results
        ]
        result_df[f"{analyzer_id}_ppl_without_instruction"] = [
            r["ppl_without_instruction"] for r in results
        ]
        result_df[f"{analyzer_id}_response_loss"] = [
            r["response_loss"] for r in results
        ]

        return result_df

    def analyze_sample(
        self,
        df: pd.DataFrame,
        schema: Optional[dict] = None,
    ) -> pd.DataFrame:
        """Analyze instruction-response pairs for IFD scores.

        This analyzer supports two data formats:
        1. Flat format: DataFrames with separate instruction and response columns
           (e.g., Alpaca format with 'instruction' and 'output' columns)
        2. Conversation format: Message-level DataFrames with 'text_content' and
           'role' columns (standard Oumi analysis pipeline)

        For conversation format, IFD is computed for each assistant message using
        the preceding user message as the instruction.

        Args:
            df: Input DataFrame with instruction/response or text_content/role.
            schema: Column schema dict (used to identify columns if not set).

        Returns:
            DataFrame with added IFD analysis columns.
        """
        result_df = df.copy()
        analyzer_id = getattr(self, "analyzer_id", "ifd")

        # Load model
        self._load_model()

        # Check if this is conversation format
        if self._is_conversation_format(df):
            logger.info(
                "Detected conversation format. Computing IFD for assistant messages "
                "using preceding user messages as instructions."
            )
            return self._analyze_conversation_format(df, analyzer_id)

        # Otherwise, try flat instruction-response format
        instruction_col, response_col = self._find_instruction_response_columns(
            df, schema
        )

        if instruction_col is None or response_col is None:
            logger.warning(
                "Could not find instruction and response columns. "
                "For flat format, set instruction_column and response_column, "
                "or use columns like 'instruction'/'prompt' and 'response'/'output'. "
                "For conversation format, ensure 'text_content' and 'role' exist. "
                f"Available columns: {list(df.columns)}"
            )
            return result_df

        logger.info(
            f"Computing IFD scores using instruction='{instruction_col}', "
            f"response='{response_col}'"
        )

        result_df = self._analyze_flat_format(
            df, instruction_col, response_col, analyzer_id
        )

        # Log summary statistics
        ifd_col = f"{analyzer_id}_score"
        ifd_scores = result_df[ifd_col].dropna().tolist()
        if ifd_scores:
            logger.info(
                f"IFD analysis complete. "
                f"Mean IFD: {sum(ifd_scores)/len(ifd_scores):.3f}, "
                f"Min: {min(ifd_scores):.3f}, Max: {max(ifd_scores):.3f}"
            )

        return result_df
