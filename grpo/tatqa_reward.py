"""TatQA reward function for GRPO training with LLM judge."""

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import dotenv

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs import InferenceConfig, InferenceEngineType
from oumi.core.registry import RegistryType, register
from oumi.core.types.conversation import Conversation, Message, Role

# Load environment variables from .env file (for API keys)
dotenv.load_dotenv()


# -----------------------------
# Answer extraction
# -----------------------------
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer(completion: str) -> str | None:
    """Extract answer from <answer>...</answer> tags.

    Args:
        completion: The completion string from the LLM

    Returns:
        The extracted answer, or None if not found
    """
    if not completion:
        return None
    match = ANSWER_RE.search(completion)
    if not match:
        return None
    ans = match.group(1).strip()
    return ans if ans else None


# -----------------------------
# Text normalization
# -----------------------------
def _normalize_answer(text: str) -> str:
    """Normalize text for comparison.

    Normalization steps:
    - Remove extra whitespace
    - Lowercase
    - Remove currency symbols ($, £, €, ¥)
    - Remove commas from numbers
    - Remove percentage signs
    - Strip leading/trailing whitespace

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Lowercase
    text = text.lower()

    # Remove currency symbols
    text = re.sub(r"[$£€¥]", "", text)

    # Remove commas from numbers (but preserve text commas)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)

    # Remove percentage signs adjacent to numbers
    text = re.sub(r"(\d)\s*%", r"\1", text)

    # Remove common units/words
    text = re.sub(r"\b(thousand|million|billion)\b", "", text)

    return text.strip()


def _normalized_equal(a: str, b: str) -> bool:
    """Check if two strings are equal after normalization.

    Args:
        a: First string
        b: Second string

    Returns:
        True if normalized strings match
    """
    return _normalize_answer(a) == _normalize_answer(b)


# -----------------------------
# LLM Judge
# -----------------------------
@dataclass
class JudgeResult:
    """Result from LLM judge."""

    decision: str  # "equal" or "not_equal"
    confidence: float | None = None
    rationale: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from text.

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed JSON object, or None if not found
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    return None
    return None


class TatqaJudgeClient:
    """LLM judge for TatQA answer equivalence."""

    def __init__(self, judge_config_file: str):
        """Initialize the judge client.

        Args:
            judge_config_file: Path to YAML config file for judge model
        """
        self.config = InferenceConfig.from_yaml(str(judge_config_file))

        # Use the inference engine builder to support multiple backends
        # Engine type can be specified in the config file (e.g., vllm, openai, remote_vllm)
        # If not specified, defaults to VLLM for open source models
        engine_type = self.config.engine or InferenceEngineType.VLLM
        self.engine = build_inference_engine(
            engine_type=engine_type,
            model_params=self.config.model,
            remote_params=self.config.remote_params,
            generation_params=self.config.generation,
        )

    def build_prompt(self, gt: str, pred: str) -> str:
        """Build judge prompt.

        Args:
            gt: Ground truth answer
            pred: Predicted answer

        Returns:
            Prompt string for the judge model
        """
        return f"""You are an equivalence judge for table-based question answering.

Determine if the predicted answer is semantically equivalent to the ground-truth answer.

Guidelines:
- For numeric answers: consider formatting (commas, currency symbols, units, signs), reasonable rounding
- For text answers: consider paraphrasing, synonyms, minor wording differences
- For dates/years: consider different formats (2019, 2019.0, "2019")
- Answers must convey the same essential information
- Ignore minor formatting differences

Return ONLY a JSON object with this schema:
{{
  "decision": "equal" or "not_equal",
  "confidence": number between 0 and 1,
  "rationale": "short explanation"
}}

Ground truth: {gt}
Prediction: {pred}
"""

    def call(self, prompt: str) -> str:
        """Call the judge model.

        Args:
            prompt: Prompt to send to judge

        Returns:
            Model response
        """
        input_data = [Conversation(messages=[Message(role=Role.USER, content=prompt)])]
        results = self.engine.infer(input=input_data, inference_config=self.config)
        return str(results[0].messages[-1].content).strip()

    def judge(self, gt: str, pred: str) -> JudgeResult:
        """Judge a single prediction.

        Args:
            gt: Ground truth answer
            pred: Predicted answer

        Returns:
            JudgeResult with decision and metadata
        """
        prompt = self.build_prompt(gt, pred)
        try:
            resp = self.call(prompt)

            # Try to parse JSON
            parsed = None
            try:
                parsed = json.loads(resp)
            except Exception:
                parsed = _extract_first_json_object(resp)

            if parsed is None:
                return JudgeResult(
                    decision="not_equal",
                    raw={"model_text": resp},
                    error="judge_response_not_json",
                )

            decision = parsed.get("decision")
            if decision not in ("equal", "not_equal"):
                return JudgeResult(
                    decision="not_equal",
                    confidence=parsed.get("confidence"),
                    rationale=parsed.get("rationale"),
                    raw={"model_text": resp, "parsed": parsed},
                    error="judge_invalid_decision",
                )

            return JudgeResult(
                decision=decision,
                confidence=parsed.get("confidence"),
                rationale=parsed.get("rationale"),
                raw={"model_text": resp, "parsed": parsed},
                error=None,
            )

        except Exception as e:
            return JudgeResult(
                decision="not_equal",
                raw=None,
                error=f"judge_call_exception: {type(e).__name__}: {e}",
            )

    def judge_batch(
        self, gt_list: list[str], pred_list: list[str]
    ) -> list[JudgeResult]:
        """Judge a batch of predictions efficiently.

        Args:
            gt_list: List of ground truth answers
            pred_list: List of predicted answers

        Returns:
            List of JudgeResult objects
        """
        if len(gt_list) != len(pred_list):
            raise ValueError("gt_list and pred_list must have same length")

        # Build all prompts
        prompts = [self.build_prompt(gt, pred) for gt, pred in zip(gt_list, pred_list)]

        # Convert to conversations
        conversations = [
            Conversation(messages=[Message(role=Role.USER, content=prompt)])
            for prompt in prompts
        ]

        # Batch inference
        try:
            results = self.engine.infer(
                input=conversations, inference_config=self.config
            )
        except Exception as e:
            # If batch inference fails, return error for all
            error_result = JudgeResult(
                decision="not_equal",
                raw=None,
                error=f"batch_inference_failed: {type(e).__name__}: {e}",
            )
            return [error_result] * len(gt_list)

        # Parse results
        judge_results = []
        for result in results:
            resp_text = str(result.messages[-1].content).strip()

            # Try to parse JSON
            parsed = None
            try:
                parsed = json.loads(resp_text)
            except Exception:
                parsed = _extract_first_json_object(resp_text)

            if parsed is None:
                judge_results.append(
                    JudgeResult(
                        decision="not_equal",
                        raw={"model_text": resp_text},
                        error="judge_response_not_json",
                    )
                )
                continue

            decision = parsed.get("decision")
            if decision not in ("equal", "not_equal"):
                judge_results.append(
                    JudgeResult(
                        decision="not_equal",
                        confidence=parsed.get("confidence"),
                        rationale=parsed.get("rationale"),
                        raw={"model_text": resp_text, "parsed": parsed},
                        error="judge_invalid_decision",
                    )
                )
                continue

            judge_results.append(
                JudgeResult(
                    decision=decision,
                    confidence=parsed.get("confidence"),
                    rationale=parsed.get("rationale"),
                    raw={"model_text": resp_text, "parsed": parsed},
                    error=None,
                )
            )

        return judge_results


# -----------------------------
# Reward function
# -----------------------------
def _is_conversational(data: Any) -> bool:
    """Check if data is in conversational format.

    Args:
        data: Data to check

    Returns:
        True if conversational format
    """
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and "role" in first and "content" in first:
            return True
    return False


# Global judge client (cached)
_judge_client_cache: dict[str, TatqaJudgeClient] = {}


@register("tatqa", RegistryType.REWARD_FUNCTION)
def tatqa_reward(
    completions: list[list[dict[str, Any]]] | list[str],
    ground_truth: list[str] | None = None,
    judge_config_file: str | None = None,
    use_judge: bool = True,
    format_score: float = 0.0,
    correct_score: float = 1.0,
    **kwargs: dict[str, Any],
) -> list[float]:
    """TatQA reward function with LLM judge.

    Scoring tiers:
    1. No answer extracted (format error): 0.0
    2. Exact match: correct_score (default 1.0)
    3. Normalized match: correct_score
    4. LLM judge says "equal": correct_score
    5. LLM judge says "not_equal": format_score (default 0.0)

    Args:
        completions: Model completions (conversational or string format)
        ground_truth: List of ground truth answers
        judge_config_file: Path to judge model config (required if use_judge=True)
        use_judge: Whether to use LLM judge for non-exact matches
        format_score: Reward for valid format but incorrect answer
        correct_score: Reward for correct answer
        kwargs: Additional keyword arguments from dataset

    Returns:
        List of reward scores
    """
    # Handle conversational format
    if completions and _is_conversational(completions[0]):
        completion_strs = [c[0]["content"] for c in completions]
    else:
        completion_strs = completions

    # Get ground truths from various sources
    ground_truths = ground_truth or kwargs.get("ground_truth")
    if ground_truths is None:
        raise ValueError(
            "No ground truth provided. Dataset must include 'ground_truth' field."
        )

    # Initialize judge client if needed
    judge_client = None
    if use_judge:
        if judge_config_file is None:
            raise ValueError(
                "judge_config_file is required when use_judge=True. "
                "Please provide path to judge model config."
            )

        # Cache judge client to avoid reloading
        if judge_config_file not in _judge_client_cache:
            _judge_client_cache[judge_config_file] = TatqaJudgeClient(judge_config_file)
        judge_client = _judge_client_cache[judge_config_file]

    # Compute rewards
    rewards = []
    judge_needed = []  # Track which samples need judge

    for i, (completion, gt) in enumerate(zip(completion_strs, ground_truths)):
        try:
            # Extract answer from completion
            pred = _extract_answer(completion)

            # No answer extracted -> format error
            if pred is None:
                rewards.append(0.0)
                judge_needed.append(None)
                continue

            # Fast path: exact match
            if pred == str(gt):
                rewards.append(correct_score)
                judge_needed.append(None)
                continue

            # Fast path: normalized match
            if _normalized_equal(pred, str(gt)):
                rewards.append(correct_score)
                judge_needed.append(None)
                continue

            # Need judge
            if use_judge and judge_client is not None:
                judge_needed.append((i, pred, str(gt)))
                rewards.append(None)  # Placeholder, will fill in later
            else:
                # No judge, treat as incorrect
                rewards.append(format_score)
                judge_needed.append(None)

        except Exception:
            # On any error, give 0.0 reward
            rewards.append(0.0)
            judge_needed.append(None)

    # Batch judge for samples that need it
    if use_judge and judge_client is not None and any(j is not None for j in judge_needed):
        # Extract samples that need judging
        judge_samples = [j for j in judge_needed if j is not None]
        indices = [j[0] for j in judge_samples]
        preds = [j[1] for j in judge_samples]
        gts = [j[2] for j in judge_samples]

        # Batch judge
        judge_results = judge_client.judge_batch(gts, preds)

        # Fill in rewards
        for idx, judge_result in zip(indices, judge_results):
            if judge_result.decision == "equal":
                rewards[idx] = correct_score
            else:
                rewards[idx] = format_score

    return rewards
