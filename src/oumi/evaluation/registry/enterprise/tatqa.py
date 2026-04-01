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

"""TatQA evaluation function with optional LLM-as-judge scoring."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from oumi.core.configs import InferenceConfig
from oumi.core.configs.params.evaluation_params import EvaluationTaskParams
from oumi.core.inference.base_inference_engine import BaseInferenceEngine
from oumi.core.registry import register_evaluation_function
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference import OpenAIInferenceEngine
from oumi.utils.logging import logger

# -----------------------------------------------------------------------------
# Answer extraction
# -----------------------------------------------------------------------------
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def _extract_answer(response_text: str) -> str | None:
    """Extract the last <answer>...</answer> span from a model response."""
    if not response_text:
        return None
    matches = [m.strip() for m in _ANSWER_RE.findall(response_text) if m.strip()]
    return matches[-1] if matches else None


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", s.strip())


def _normalized_equal(a: str, b: str) -> bool:
    """Case- and whitespace-insensitive equality check."""
    return _normalize_text(a).lower() == _normalize_text(b).lower()


# -----------------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------------
def _load_test_data(test_data_path: str) -> list[dict]:
    """Load test data from a JSONL file."""
    data = []
    with open(test_data_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _extract_ground_truth(conversation: dict) -> str:
    """Return the ground-truth answer for a conversation.

    Checks ``metadata.ground_truth`` first (the format used by TatQA test data),
    then falls back to the assistant message content for compatibility with
    other enterprise data formats.
    """
    gt = conversation.get("metadata", {}).get("ground_truth")
    if gt is not None:
        return str(gt).strip()
    for msg in conversation.get("messages", []):
        if msg.get("role") == "assistant":
            return msg.get("content", "").strip()
    return ""


def _create_input_conversation(conversation: dict) -> Conversation:
    """Build an inference-ready Conversation from system + user turns."""
    messages = []
    for msg in conversation.get("messages", []):
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            messages.append(Message(role=Role.SYSTEM, content=content))
        elif role == "user":
            messages.append(Message(role=Role.USER, content=content))
            break  # stop after first user turn
    return Conversation(messages=messages)


# -----------------------------------------------------------------------------
# LLM judge (internal)
# -----------------------------------------------------------------------------
@dataclass
class _JudgeResult:
    decision: str  # "equal" | "not_equal"
    confidence: float | None = None
    rationale: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of an arbitrary string."""
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


class _TatQAJudge:
    """Thin wrapper around an OpenAI-compatible inference engine used as a judge."""

    _JUDGE_PROMPT_TEMPLATE = """\
You are an equivalence judge for QA answers.

Determine if the predicted answer is semantically equivalent to the ground-truth answer.
- Consider numeric formatting, commas, currency symbols, rounding if clearly equivalent.
- Consider trivial rephrasing equivalent.
- If the predicted answer is missing critical information or changes meaning, it is NOT equivalent.

Return ONLY a JSON object with this schema:
{{
  "decision": "equal" or "not_equal",
  "confidence": number between 0 and 1,
  "rationale": "short explanation"
}}

Ground truth: {gt}
Prediction: {pred}
"""

    def __init__(self, judge_config_file: str):
        self.config = InferenceConfig.from_yaml(judge_config_file)
        self.engine = OpenAIInferenceEngine(
            model_params=self.config.model,
            remote_params=self.config.remote_params,
            generation_params=self.config.generation,
        )

    def build_prompt(self, gt: str, pred: str) -> str:
        return self._JUDGE_PROMPT_TEMPLATE.format(gt=gt, pred=pred)


# -----------------------------------------------------------------------------
# Batched evaluation loop
# -----------------------------------------------------------------------------
def _run_evaluation(
    ground_truths: list[str],
    raw_responses: list[str],
    judge: "_TatQAJudge | None",
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Evaluate predictions against ground truths.

    Uses a three-tier cascade:
      1. Exact string match
      2. Normalized (whitespace/case) match
      3. Batched LLM judge call (only when judge is provided and tiers 1-2 fail)

    Returns:
        A tuple of (counts_dict, per_example_log_list).
    """
    equal_count = 0
    mismatch_count = 0
    format_error_count = 0
    judge_called_count = 0
    judge_error_count = 0

    logs: list[dict[str, Any]] = []

    # --- Pass 1: exact / normalized checks; collect items that need judging ---
    judge_inputs: list[Conversation] = []
    # (log_position, sample_idx, gt_str, raw_content, pred)
    judge_meta: list[tuple[int, int, str, str, str]] = []

    for sample_idx, (gt, raw_content) in enumerate(zip(ground_truths, raw_responses)):
        gt_str = str(gt)
        pred = _extract_answer(raw_content)

        if pred is None:
            format_error_count += 1
            logs.append(
                {
                    "idx": sample_idx,
                    "gt": gt_str,
                    "raw_content": raw_content,
                    "pred_extracted": None,
                    "status": "format_error",
                    "method": "none",
                    "judge": None,
                }
            )
            continue

        if pred == gt_str:
            equal_count += 1
            logs.append(
                {
                    "idx": sample_idx,
                    "gt": gt_str,
                    "raw_content": raw_content,
                    "pred_extracted": pred,
                    "status": "equal",
                    "method": "exact",
                    "judge": None,
                }
            )
            continue

        if _normalized_equal(pred, gt_str):
            equal_count += 1
            logs.append(
                {
                    "idx": sample_idx,
                    "gt": gt_str,
                    "raw_content": raw_content,
                    "pred_extracted": pred,
                    "status": "equal",
                    "method": "normalized",
                    "judge": None,
                }
            )
            continue

        # Needs judging (or falls through to mismatch if no judge)
        if judge is not None:
            log_pos = len(logs)
            prompt = judge.build_prompt(gt_str, pred)
            judge_inputs.append(
                Conversation(messages=[Message(role=Role.USER, content=prompt)])
            )
            judge_meta.append((log_pos, sample_idx, gt_str, raw_content, pred))
            logs.append({})  # placeholder, filled in pass 2
        else:
            mismatch_count += 1
            logs.append(
                {
                    "idx": sample_idx,
                    "gt": gt_str,
                    "raw_content": raw_content,
                    "pred_extracted": pred,
                    "status": "mismatch",
                    "method": "none",
                    "judge": None,
                }
            )

    # --- Pass 2: batched judge call ---
    if judge_inputs and judge is not None:
        judge_called_count = len(judge_inputs)
        logger.info(f"Calling judge on {judge_called_count} non-matching examples...")
        judge_outputs = judge.engine.infer(
            input=judge_inputs, inference_config=judge.config
        )

        for out, (log_pos, sample_idx, gt_str, raw_content, pred) in zip(
            judge_outputs, judge_meta
        ):
            resp_text = str(out.messages[-1].content).strip()

            parsed: dict[str, Any] | None = None
            try:
                parsed = json.loads(resp_text)
            except Exception:
                parsed = _extract_first_json_object(resp_text)

            if parsed is None:
                judge_error_count += 1
                mismatch_count += 1
                jr = _JudgeResult(
                    decision="not_equal",
                    raw={"model_text": resp_text},
                    error="judge_response_not_json",
                )
                logs[log_pos] = {
                    "idx": sample_idx,
                    "gt": gt_str,
                    "raw_content": raw_content,
                    "pred_extracted": pred,
                    "status": "mismatch",
                    "method": "judge",
                    "judge": asdict(jr),
                }
                continue

            decision = parsed.get("decision")
            if decision == "equal":
                equal_count += 1
                status = "equal"
                err = None
            elif decision == "not_equal":
                mismatch_count += 1
                status = "mismatch"
                err = None
            else:
                mismatch_count += 1
                status = "mismatch"
                err = "judge_invalid_decision"
                judge_error_count += 1

            jr = _JudgeResult(
                decision=decision
                if decision in ("equal", "not_equal")
                else "not_equal",
                confidence=parsed.get("confidence"),
                rationale=parsed.get("rationale"),
                raw={"model_text": resp_text, "parsed": parsed},
                error=err,
            )
            logs[log_pos] = {
                "idx": sample_idx,
                "gt": gt_str,
                "raw_content": raw_content,
                "pred_extracted": pred,
                "status": status,
                "method": "judge",
                "judge": asdict(jr),
            }

    total = len(logs)
    counts = {
        "equal": equal_count,
        "mismatch": mismatch_count,
        "format_error": format_error_count,
        "judge_called": judge_called_count,
        "judge_errors": judge_error_count,
        "total": total,
    }
    return counts, logs


# -----------------------------------------------------------------------------
# Registered evaluation function
# -----------------------------------------------------------------------------
@register_evaluation_function("enterprise_tatqa")
def enterprise_tatqa(
    task_params: EvaluationTaskParams,
    inference_engine: BaseInferenceEngine,
    test_data_path: str = "data/enterprise/tatqa/test.jsonl",
    judge_config_file: str | None = None,
) -> dict[str, Any]:
    """Evaluate TatQA financial/tabular QA with optional LLM-as-judge scoring.

    Expects a JSONL file where each record has:

    - ``messages``: list of ``{role, content}`` dicts (system + user turns).
    - ``metadata.ground_truth``: the expected answer string.

    The model is expected to output ``<answer>...</answer>`` tags.  Scoring uses
    a three-tier cascade:

    1. Exact string match
    2. Whitespace/case-normalised match
    3. LLM judge (batched) — only when ``judge_config_file`` is provided

    Args:
        task_params: Evaluation task parameters (e.g. ``num_samples``).
        inference_engine: Engine used to generate model predictions.
        test_data_path: Path to the test JSONL file.
        judge_config_file: Optional path to an ``InferenceConfig`` YAML for the
            judge LLM (must be OpenAI-compatible).  When omitted, only tiers 1-2
            are used.

    Returns:
        Dictionary of metrics plus a ``_predictions`` key saved separately by
        the evaluator framework.
    """
    if not Path(test_data_path).exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Ensure the file exists before running this evaluator."
        )

    logger.info(f"Loading TatQA test data from {test_data_path}")
    test_data = _load_test_data(test_data_path)

    num_samples = task_params.num_samples
    if num_samples is not None and num_samples > 0:
        test_data = test_data[:num_samples]

    logger.info(f"Evaluating on {len(test_data)} samples")

    ground_truths = [_extract_ground_truth(conv) for conv in test_data]
    input_conversations = [_create_input_conversation(conv) for conv in test_data]

    logger.info("Running inference...")
    output_conversations = inference_engine.infer(input_conversations)

    raw_responses: list[str] = []
    for conv in output_conversations:
        msg = conv.last_message()
        raw_responses.append(
            msg.content if (msg and isinstance(msg.content, str)) else ""
        )

    judge: _TatQAJudge | None = None
    if judge_config_file:
        logger.info(f"Initialising judge from {judge_config_file}")
        judge = _TatQAJudge(judge_config_file=judge_config_file)

    counts, logs = _run_evaluation(ground_truths, raw_responses, judge)

    total = counts["total"]
    equal = counts["equal"]
    format_errors = counts["format_error"]
    judge_called = counts["judge_called"]
    judge_errors = counts["judge_errors"]

    exact_correct = sum(1 for lg in logs if lg.get("method") == "exact")
    normalized_correct = sum(1 for lg in logs if lg.get("method") == "normalized")
    judge_correct = sum(
        1 for lg in logs if lg.get("method") == "judge" and lg.get("status") == "equal"
    )

    mean_response_chars = (
        sum(len(r) for r in raw_responses) / len(raw_responses)
        if raw_responses
        else 0.0
    )

    metrics: dict[str, Any] = {
        "accuracy": equal / total if total > 0 else 0.0,
        "exact_match_rate": exact_correct / total if total > 0 else 0.0,
        "normalized_match_rate": normalized_correct / total if total > 0 else 0.0,
        "judge_correct_rate": judge_correct / total if total > 0 else 0.0,
        "format_error_rate": format_errors / total if total > 0 else 0.0,
        "judge_called": judge_called,
        "judge_error_rate": judge_errors / judge_called if judge_called > 0 else 0.0,
        "mean_response_chars": mean_response_chars,
        "num_correct": equal,
        "num_total": total,
    }

    # Build _predictions in oumi conversation format
    metrics["_predictions"] = [
        {
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content or "",
                }
                for m in input_conversations[lg["idx"]].messages
            ]
            + [{"role": "assistant", "content": lg.get("raw_content", "")}],
            "metadata": {
                "ground_truth": lg.get("gt", ""),
                "pred_extracted": lg.get("pred_extracted"),
                "status": lg.get("status", ""),
                "method": lg.get("method", ""),
                "judge": lg.get("judge"),
            },
        }
        for lg in logs
    ]

    logger.info(
        f"TatQA results: accuracy={metrics['accuracy']:.4f} "
        f"({equal}/{total}) | "
        f"exact={exact_correct}, normalized={normalized_correct}, "
        f"judge={judge_correct}, format_err={format_errors}"
    )

    return metrics
