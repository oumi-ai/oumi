from collections import namedtuple

BenchmarkConfig = namedtuple("BenchmarkConfig", ["name", "num_fewshot", "num_samples"])

HUGGINGFACE_LEADERBOARD_V1 = "huggingface_leaderboard_v1"

BENCHMARK_CONFIGS = {
    HUGGINGFACE_LEADERBOARD_V1: [
        BenchmarkConfig("mmlu", 5, None),
        BenchmarkConfig("arc_challenge", 25, None),
        BenchmarkConfig("winogrande", 5, None),
        BenchmarkConfig("hellaswag", 10, None),
        BenchmarkConfig("truthfulqa_mc2", 0, None),
        # BenchmarkConfig("gsm8k", 5, None),  # Temporarily removed due to runtime 2h10m
    ],
}
