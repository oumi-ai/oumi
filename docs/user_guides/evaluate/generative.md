# Generative Evaluations

Evaluating generative language models requires specialized approaches beyond traditional metrics. Here we cover several established methods for assessing model performance.

All evaluations in Oumi are automatically logged and versioned, capturing model configurations, evaluation parameters, and environmental details to ensure reproducible results.


## Established Benchmarks

### AlpacaEval

AlpacaEval is a framework for evaluating instruction-following capabilities of language models. It uses GPT-4 as a judge to compare model outputs against reference responses.

AlpacaEval provides automated evaluation capabilities using GPT-4 as a judge to assess instruction-following abilities of language models. The framework focuses specifically on measuring how well models follow instructions by comparing their outputs against reference responses.

It generates standardized win rates against baseline models and has been widely adopted as a benchmark in research papers. This makes it particularly useful for evaluating instruction-tuned models, comparing performance against established baselines, and conducting automated evaluations at scale.

To use AlpacaEval, you can run the following command:

```bash
oumi evaluate -c configs/examples/evaluation/alpaca/alpaca_v2.yaml
```

You can also run the example notebook {gh}`notebooks/Oumi - Evaluation with AlpacaEval 2.0.ipynb`.

**Resources:**
- {gh}`Evaluation Tutorial <notebooks/Oumi - Evaluation with AlpacaEval 2.0.ipynb>`
- [AlpacaEval Paper](https://arxiv.org/abs/2305.14387)
- [Official Repository](https://github.com/tatsu-lab/alpaca_eval)

### MT-Bench

MT-Bench (Multi-Turn Benchmark) is an evaluation framework specifically designed for assessing chat assistants in multi-turn conversations. It tests models' abilities to maintain context, provide consistent responses across turns, and engage in coherent dialogues.

MT-Bench offers several key features including multi-turn conversation evaluation with increasing complexity, diverse question categories spanning various domains, and a standardized scoring system powered by GPT-4 judgments.

To evaluate a model with MT-Bench, see the example notebook {gh}`Oumi - Evaluation with MT Bench.ipynb`.


**Resources:**
- {gh}`MT-Bench Tutorial <notebooks/Oumi - Evaluation with MT Bench.ipynb>`
- [MT-Bench Paper](https://arxiv.org/abs/2306.05685)
- [Official Implementation](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)
- [Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

### HumanEval

HumanEval is a benchmark designed to evaluate language models' capabilities in generating functional code from natural language descriptions. It consists of programming challenges that test both understanding of requirements and ability to generate correct, efficient code solutions.

**Resources:**
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [Official Repository](https://github.com/openai/human-eval)
- [Dataset Documentation](https://huggingface.co/datasets/openai_humaneval)

## LLM-as-a-judge

Using LLMs as judges provides a flexible approach to evaluate model outputs.

Oumi supports various judge configurations and scoring criteria. See {doc}`/user_guides/judge/judge` to get started.

**Resources:**
- {gh}`Custom Judge Implementation <notebooks/Oumi - Custom Judge.ipynb>`
- {gh}`Oumi Judge Framework <notebooks/Oumi - Oumi Judge.ipynb>`


## Results and Logging

All evaluation results are automatically saved and can be tracked:

- **Local Results**: Saved in the specified `output_dir` with detailed metrics, configurations, and version information
- **Weights & Biases**: When enabled, results are automatically logged along with model configurations, generation parameters, and environmental details.

For more details on configuration options and advanced usage, see the {doc}`/user_guides/evaluate/evaluate` documentation.
