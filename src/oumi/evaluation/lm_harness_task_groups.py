from oumi.core.configs import LMHarnessParams

HUGGINGFACE_LEADERBOARD_V1 = "huggingface_leaderboard_v1"

TASK_GROUPS = {
    HUGGINGFACE_LEADERBOARD_V1: [
        LMHarnessParams(tasks=["mmlu"], num_fewshot=5, num_samples=None),
        LMHarnessParams(tasks=["arc_challenge"], num_fewshot=25, num_samples=None),
        LMHarnessParams(tasks=["winogrande"], num_fewshot=5, num_samples=None),
        LMHarnessParams(tasks=["hellaswag"], num_fewshot=10, num_samples=None),
        LMHarnessParams(tasks=["truthfulqa_mc2"], num_fewshot=0, num_samples=None),
        LMHarnessParams(tasks=["gsm8k"], num_fewshot=5, num_samples=None),
    ],
}
