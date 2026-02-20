import re
from typing import Any, Optional
import ast
from collections import Counter
from fractions import Fraction
from typing import Optional

from oumi.core.registry import RegistryType, register

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def is_conversational(messages):
    if isinstance(messages, list):
        message = messages[0]
        # Each message must a list of dictionaries with keys "role" and "content"
        if isinstance(message, dict) and "role" in message and "content" in message:
            return True
    return False

def normalize_ops(s: str) -> str:
    """Normalize common unicode operator variants to ASCII."""
    return (
        s.replace("×", "*")
         .replace("÷", "/")
         .replace("−", "-")
         .strip()
    )

def extract_final_answer_text(completion: str) -> Optional[str]:
    """
    Return the inner string of the (single) <answer>...</answer>.
    Returns None if missing OR if there is not exactly one answer block.
    Adds synthetic <think> prefix to match your current behavior.
    """
    completion = "<think>" + completion
    matches = ANSWER_RE.findall(completion or "")
    if len(matches) == 0:
        return None
    ans = normalize_ops(matches[-1])
    return ans if ans else None

def split_lhs_rhs(answer_text: str) -> tuple[str, Optional[str]]:
    """
    If '=' is present, treat as equation and split on the LAST '='.
    """
    if "=" not in answer_text:
        return answer_text.strip(), None
    lhs, rhs = answer_text.rsplit("=", 1)
    return lhs.strip(), rhs.strip()

class SafeCountdownEvaluator:
    """
    Safe evaluator for arithmetic expressions using only +, -, *, / and parentheses.
    - Uses Fractions for exact arithmetic.
    - Tracks integer literals encountered (for number-usage checking).
    """
    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def __init__(self) -> None:
        self.used_numbers: list[int] = []

    def parse(self, expr: str) -> ast.AST:
        return ast.parse(expr, mode="eval")

    def eval(self, node: ast.AST) -> Fraction:
        return self._eval_node(node)

    def _eval_node(self, node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or node.value is None:
                raise ValueError("Invalid literal")
            if isinstance(node.value, int):
                self.used_numbers.append(int(node.value))
                return Fraction(int(node.value), 1)
            if isinstance(node.value, float):
                raise ValueError("Float literals not allowed; use integers only")
            raise ValueError("Unsupported literal type")

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, self.ALLOWED_UNARYOPS):
            val = self._eval_node(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val

        if isinstance(node, ast.BinOp) and isinstance(node.op, self.ALLOWED_BINOPS):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero")
                return left / right

        raise ValueError(f"Disallowed syntax: {type(node).__name__}")

def eval_expr(expr: str) -> tuple[Fraction, list[int]]:
    ev = SafeCountdownEvaluator()
    tree = ev.parse(expr)
    val = ev.eval(tree)
    return val, ev.used_numbers

def check_number_usage_exactly_once(used_nums: list[int], provided_nums: list[int]) -> bool:
    """
    Exact rule: LHS must use ALL provided numbers exactly once.
    This is exact multiset equality.
    """
    return Counter(used_nums) == Counter(provided_nums)

def compute_format_reward_func(completions, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
      
      Returns:
          list[float]: Reward scores
    """
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]

    rewards = []
    for completion in completions:
 
      try:
        # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
        completion = "<think>" + completion        
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
 
        match = re.search(regex, completion, re.DOTALL) 
        # if the format is not correct, reward is 0
        if match is None or len(match.groups()) != 2:
            rewards.append(0.0)
        else:
            rewards.append(1.0)
      except Exception:
        rewards.append(0.0)
    return rewards
 
def compute_numbers_rule_reward(completions, **kwargs):
    """
    +1 if LHS uses ALL provided nums exactly once. Otherwise 0.
    - Requires exactly one <answer>...</answer>
    - If '=' exists, ignores RHS numbers for usage.
    """

    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]

    num_ls = kwargs["nums"]

    rewards = []
    for completion, provided_numbers in zip(completions, num_ls):
        try:
            ans = extract_final_answer_text(completion)
            if ans is None:
                rewards.append(0.0)
                continue

            lhs, _rhs = split_lhs_rhs(ans)

            # Evaluate LHS to collect used numbers (AST-safe)
            _lhs_val, used_nums = eval_expr(lhs)

            rewards.append(1.0 if check_number_usage_exactly_once(used_nums, provided_numbers) else 0.0)

        except Exception:
            rewards.append(0.0)

    return rewards

def compute_correct_equation_reward_func(completions, **kwargs):
    """
    +1 if equation is correct AND uses ALL numbers exactly once on LHS. Otherwise 0.
    Rules:
      - at least one <answer>...</answer>, extracts the final answer
      - LHS uses all numbers exactly once
      - if '=' present: require LHS == RHS == target (ignore RHS number usage)
      - else: require LHS == target
    """
    if is_conversational(completions[0]):
        completions = [completion[0]["content"] for completion in completions]

    target_ls = kwargs["target"]
    num_ls = kwargs["nums"]
    rewards = []
    for completion, target, nums in zip(completions, target_ls, num_ls):
        try:
            ans = extract_final_answer_text(completion)
            if ans is None:
                rewards.append(0.0)
                continue

            lhs, rhs = split_lhs_rhs(ans)
            lhs_val, used_nums = eval_expr(lhs)
            target_frac = Fraction(int(target), 1)

            if rhs is not None and rhs != "":
                rhs_val, _ = eval_expr(rhs)  # ignore rhs usage
                print(lhs_val, rhs_val, target_frac)
                rewards.append(1.0 if (lhs_val == rhs_val == target_frac) else 0.0)
            else:
                print(lhs_val, target_frac)
                rewards.append(1.0 if (lhs_val == target_frac) else 0.0)

        except Exception:
            rewards.append(0.0)

    return rewards


@register("format_reward", RegistryType.REWARD_FUNCTION)
def _format_reward(
    completions: list[list[dict[str, Any]]],
    **kwargs: dict[str, Any],
) -> list[float]:
    """Custom reward function for counting letters in a string.

    For more details on custom reward functions used in trl's GRPOTrainer, see:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function.

    Args:
        completions: 
        kwargs: 

    Returns:
    """
    return compute_format_reward_func(completions, **kwargs)

@register("equation_reward", RegistryType.REWARD_FUNCTION)
def _equation_reward(
    completions: list[list[dict[str, Any]]],
    **kwargs: dict[str, Any],
) -> list[float]:
    """Custom reward function for counting letters in a string.

    For more details on custom reward functions used in trl's GRPOTrainer, see:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function.

    Args:
        completions: 
        kwargs: 

    Returns:
    """
    return compute_correct_equation_reward_func(completions, **kwargs)

@register("numbers_rule_reward", RegistryType.REWARD_FUNCTION)
def _numbers_rule_reward(
    completions: list[list[dict[str, Any]]],
    **kwargs: dict[str, Any],
) -> list[float]:
    """Custom reward function for counting letters in a string.

    For more details on custom reward functions used in trl's GRPOTrainer, see:
    https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function.

    Args:
        completions: 
        kwargs: 

    Returns:
    """
    return compute_numbers_rule_reward(completions, **kwargs)