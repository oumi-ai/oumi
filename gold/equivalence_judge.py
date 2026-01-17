import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import dotenv
from tqdm import tqdm

from oumi.core.configs import InferenceConfig
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.inference import OpenAIInferenceEngine

dotenv.load_dotenv()

# -----------------------------
# 1) Answer extraction
# -----------------------------
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(response_text: str) -> str | None:
    if not response_text:
        return None
    m = ANSWER_RE.search(response_text)
    if not m:
        return None
    ans = m.group(1).strip()
    return ans if ans else None


# -----------------------------
# 2) Simple normalization
# -----------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def normalized_equal(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


# -----------------------------
# 3) LLM Judge interface
# -----------------------------
@dataclass
class JudgeResult:
    decision: str  # "equal" or "not_equal"
    confidence: float | None = None
    rationale: str | None = None
    raw: dict[str, Any] | None = None
    error: str | None = None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
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


class LLMJudgeClient:
    def __init__(self, judge_config_file: str):
        self.config = InferenceConfig.from_yaml(str(judge_config_file))

        self.engine = OpenAIInferenceEngine(
            model_params=self.config.model,
            remote_params=self.config.remote_params,
            generation_params=self.config.generation,
        )

    def build_prompt(self, gt: str, pred: str) -> str:
        return f"""You are an equivalence judge for QA answers.

Determine if the predicted answer is semantically equivalent to the ground-truth answer.
- Consider numeric formatting, commas, currency symbols, rounding if clearly equivalent.
- Consider trivial rephrasing equivalent.
- If the predicted answer is missing critical information or changes meaning, it's NOT equivalent.

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
        input_data = [Conversation(messages=[Message(role=Role.USER, content=prompt)])]
        results = self.engine.infer(input=input_data, inference_config=self.config)
        return str(results[0].messages[-1].content).strip()

    def judge(self, gt: str, pred: str) -> JudgeResult:
        prompt = self.build_prompt(gt, pred)
        try:
            resp = self.call(prompt)

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


# -----------------------------
# 4) Evaluation loop
# -----------------------------
@dataclass
class ExampleLog:
    idx: int
    gt: str
    raw_content: str
    pred_extracted: str | None
    status: str  # "equal" | "mismatch" | "format_error"
    method: str  # "exact" | "normalized" | "judge" | "none"
    judge: dict[str, Any] | None = None


def evaluate(
    ground_truth_ls,
    model_output_ls,
    judge_client: LLMJudgeClient | None,
    use_judge: bool,
) -> dict[str, Any]:
    equal_count = 0
    mismatch_count = 0
    format_error_count = 0
    judge_called_count = 0
    judge_error_count = 0

    logs = []

    for idx, (gt, response) in tqdm(
        enumerate(zip(ground_truth_ls, model_output_ls)),
        total=len(ground_truth_ls),
        desc="Evaluating",
    ):
        gt_str = str(gt)
        content = response["messages"][-1]["content"]
        pred = extract_answer(content)

        if pred is None:
            format_error_count += 1
            logs.append(
                ExampleLog(
                    idx=idx,
                    gt=gt_str,
                    raw_content=content,
                    pred_extracted=None,
                    status="format_error",
                    method="none",
                    judge=None,
                )
            )
            continue

        # fast-path equality checks
        if pred == gt_str:
            equal_count += 1
            logs.append(ExampleLog(idx, gt_str, content, pred, "equal", "exact"))
            continue

        if normalized_equal(pred, gt_str):
            equal_count += 1
            logs.append(ExampleLog(idx, gt_str, content, pred, "equal", "normalized"))
            continue

        # judge
        if use_judge and judge_client is not None:
            judge_called_count += 1
            jr = judge_client.judge(gt_str, pred)
            if jr.error:
                judge_error_count += 1

            if jr.decision == "equal":
                equal_count += 1
                status = "equal"
            else:
                mismatch_count += 1
                status = "mismatch"

            logs.append(
                ExampleLog(
                    idx=idx,
                    gt=gt_str,
                    raw_content=content,
                    pred_extracted=pred,
                    status=status,
                    method="judge",
                    judge=asdict(jr),
                )
            )
        else:
            mismatch_count += 1
            logs.append(ExampleLog(idx, gt_str, content, pred, "mismatch", "none"))

    summary = {
        "total": len(logs),
        "equal": equal_count,
        "mismatch": mismatch_count,
        "format_error": format_error_count,
        "judge_called": judge_called_count,
        "judge_errors": judge_error_count,
    }
    return {"summary": summary, "logs": logs}


def evaluate_batched(
    ground_truth_ls,
    model_output_ls,
    judge_client: LLMJudgeClient,
) -> dict[str, Any]:
    equal_count = 0
    mismatch_count = 0
    format_error_count = 0
    judge_called_count = 0
    judge_error_count = 0

    logs: list[ExampleLog] = []

    # 1) Build judge-ready inputs in one big list
    judge_inputs: list[Conversation] = []
    judge_meta: list[tuple[int, str, str, str]] = []
    # stores (idx, gt_str, raw_content, pred)

    for idx, (gt, response) in tqdm(
        enumerate(zip(ground_truth_ls, model_output_ls)),
        total=len(ground_truth_ls),
        desc="Preparing judge inputs",
    ):
        gt_str = str(gt)
        content = response["messages"][-1]["content"]
        pred = extract_answer(content)

        if pred is None:
            format_error_count += 1
            logs.append(
                ExampleLog(
                    idx=idx,
                    gt=gt_str,
                    raw_content=content,
                    pred_extracted=None,
                    status="format_error",
                    method="none",
                    judge=None,
                )
            )
            continue

        prompt = judge_client.build_prompt(gt_str, pred)
        judge_inputs.append(
            Conversation(messages=[Message(role=Role.USER, content=prompt)])
        )
        judge_meta.append((idx, gt_str, content, pred))

    # 2) Single batched call to judge engine
    if judge_inputs:
        judge_called_count = len(judge_inputs)
        results = judge_client.engine.infer(
            input=judge_inputs, inference_config=judge_client.config
        )

        # 3) Parse results and fill logs
        for out, meta in tqdm(
            zip(results, judge_meta),
            total=len(judge_meta),
            desc="Parsing judge outputs",
        ):
            idx, gt_str, content, pred = meta
            resp_text = str(out.messages[-1].content).strip()

            parsed = None
            try:
                parsed = json.loads(resp_text)
            except Exception:
                parsed = _extract_first_json_object(resp_text)

            if parsed is None:
                judge_error_count += 1
                mismatch_count += 1
                logs.append(
                    ExampleLog(
                        idx=idx,
                        gt=gt_str,
                        raw_content=content,
                        pred_extracted=pred,
                        status="mismatch",
                        method="judge",
                        judge=asdict(
                            JudgeResult(
                                decision="not_equal",
                                raw={"model_text": resp_text},
                                error="judge_response_not_json",
                            )
                        ),
                    )
                )
                continue

            decision = parsed.get("decision")
            if decision == "equal":
                equal_count += 1
                status = "equal"
                err = None
            else:
                mismatch_count += 1
                status = "mismatch"
                err = (
                    None
                    if decision in ("not_equal", "equal")
                    else "judge_invalid_decision"
                )

            if err is not None:
                judge_error_count += 1

            logs.append(
                ExampleLog(
                    idx=idx,
                    gt=gt_str,
                    raw_content=content,
                    pred_extracted=pred,
                    status=status,
                    method="judge",
                    judge=asdict(
                        JudgeResult(
                            decision=decision
                            if decision in ("equal", "not_equal")
                            else "not_equal",
                            confidence=parsed.get("confidence"),
                            rationale=parsed.get("rationale"),
                            raw={"model_text": resp_text, "parsed": parsed},
                            error=err,
                        )
                    ),
                )
            )

    summary = {
        "total": len(logs),
        "equal": equal_count,
        "mismatch": mismatch_count,
        "format_error": format_error_count,
        "judge_called": judge_called_count,
        "judge_errors": judge_error_count,
    }
    return {"summary": summary, "logs": logs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_file", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--judge_config_file", type=str, required=False)
    parser.add_argument("--score_file", type=str, required=False)
    parser.add_argument("--out_jsonl", type=str, default="eval_log.jsonl")
    parser.add_argument("--use_judge", action="store_true")
    args = parser.parse_args()

    ground_truth_ls = [
        json.loads(line)["metadata"]["ground_truth"]
        for line in open(args.ground_truth_file)
    ]
    model_output_ls = [json.loads(line) for line in open(args.results_file)]

    judge_client = None
    if args.use_judge:
        if not args.judge_config_file:
            raise ValueError("--judge_config_file is required when --use_judge is set")
        judge_client = LLMJudgeClient(judge_config_file=args.judge_config_file)
        result = evaluate_batched(ground_truth_ls, model_output_ls, judge_client)
    else:
        print("Skipping evaluation")

    if args.score_file:
        with open(args.score_file, "w") as f:
            f.write(json.dumps(result["summary"], indent=2))

    with open(args.out_jsonl, "w") as f:
        for item in result["logs"]:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
