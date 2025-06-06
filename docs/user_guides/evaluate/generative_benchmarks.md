# Generative Benchmarks

Evaluating language models on open-ended generation tasks requires specialized approaches beyond standardized benchmarks and traditional metrics. In this section, we discuss several established methods for assessing model performance with generative benchmarks. Generative benchmarks consist of open-ended questions, allowing the model to generate a free-form output, rather than adhere to a pre-defined "correct" answer. The focus of such evaluations is to assess the model's ability to follow the prompt instructions and generate human-like, high-quality, coherent, and creative responses.

## Overview

Generative benchmarks are vital to evaluate conversational agents, as well as tasks such as creative writing or editing (storytelling, essays and articles), summarization, translation, planning, and code generation. In addition, assessing capabilities such as instruction following, safety, trust, and groundedness, require generative responses.

That said, generative evaluations are significantly more challenging than closed-form evaluations, due to lack of a clear "correct" answer. This makes the evaluation criteria subjective to human judgment. But, even for an established set of criteria, aligning across raters ultimately depends on human perception, making consistent evaluations a very hard problem. Alternatively, fully-automating the rating process, by leveraging LLMs as judges of responses is recently getting more traction. LLM-as-a-judge platforms are significantly more cost- and time-effective, while they can provide reproducible and consistent results (under certain conditions).

This section discusses the LLM-as-a-judge platforms that Oumi is using as its backend to provide reliable insights on generative model performance. The evaluation process consists of 2 steps: inference and judgement. Inference generates model responses for a predefined set of open-ended prompts, while judgement leverages an LLM to judge the quality of these responses. Oumi enables generative evaluation by integrating with popular platforms (AlpacaEval and MT-Bench), as well as offering a flexible framework (see {doc}`/user_guides/judge/judge`) for users to develop their own generative evaluations.

All evaluations in Oumi are automatically logged and versioned, capturing model configurations, evaluation parameters, and environmental details to ensure reproducible results.

## Supported Out-of-the-box Benchmarks

### AlpacaEval (1.0 and 2.0)

[AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) is a framework for automatically evaluating the instruction-following capabilities of language models, as well as whether their responses are helpful, accurate, and relevant. The framework prioritizes human-aligned evaluation, aiming to assess whether the model’s response meets the expectations of human evaluators. The instruction set consists of 805 open-ended questions, such as "How did US states get their names?".

The latest update (2.0) uses GPT-4 Turbo as a judge, comparing the model outputs against a set of reference responses, and calculating standardized win-rates against these responses. AlpacaEval 2.0 has been widely adopted as a benchmark in research papers and it is particularly useful for evaluating instruction-tuned models, comparing performance against established baselines (see [leaderboard](https://tatsu-lab.github.io/alpaca_eval/)), and conducting automated evaluations at scale.

To use AlpacaEval, you can run the following command:

```bash
OPENAI_API_KEY="your_key"
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_alpaca_v2_eval.yaml
```

If you prefer to use AlpacaEval outside Oumi, we refer you to our example notebook {gh}`notebooks/Oumi - Evaluation with AlpacaEval 2.0.ipynb`.

**Resources:**
- [AlpacaEval V1.0 Paper](https://arxiv.org/abs/2305.14387)
- [AlpacaEval V2.0 Paper](https://arxiv.org/abs/2404.04475)
- [AlpacaEval V2.0 Dataset](https://huggingface.co/datasets/tatsu-lab/alpaca_eval)
- [Leaderboard](https://tatsu-lab.github.io/alpaca_eval/)
- [Official Repository](https://github.com/tatsu-lab/alpaca_eval)

### MT-Bench

[MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge) (Multi-Turn Benchmark) is an evaluation framework specifically designed for assessing chat assistants in multi-turn conversations. It tests a model's ability to maintain context, provide consistent responses across turns, and engage in coherent dialogues. The instruction set consists of 80 open-ended multi-turn questions, which span across 8 popular categories: writing, roleplay, extraction, reasoning, math, coding, STEM knowledge, knowledge of social sciences.

MT-Bench uses GPT-4 as a judge to score each answer on a scale of 10, or perform pairwise scoring between 2 models, and calculates standardized win-rates. It can also breakdown the scoring per category (see [notebook](https://colab.research.google.com/drive/15O3Y8Rxq37PuMlArE291P4OC6ia37PQK#scrollTo=5i8R0l-XqkgO)). Overall, it offers several key features including multi-turn conversation evaluation with increasing complexity, diverse question categories spanning various domains, and a standardized scoring system powered by GPT-4 judgments.

To evaluate a model with MT-Bench, see the example notebook {gh}`notebooks/Oumi - Evaluation with MT Bench.ipynb`.

**Resources:**
- {gh}`MT-Bench Tutorial <notebooks/Oumi - Evaluation with MT Bench.ipynb>`
- [MT-Bench Paper](https://arxiv.org/abs/2306.05685)
- [Human Annotation Dataset](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- [Leaderboard](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)
- [Official Repository](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)

<!--- Commented; we do NOT support HumanEval yet.
### HumanEval

HumanEval is a benchmark designed to evaluate language models' capabilities in generating functional code from natural language descriptions. It consists of programming challenges that test both understanding of requirements and ability to generate correct, efficient code solutions.

**Resources:**
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [Official Repository](https://github.com/openai/human-eval)
- [Dataset Documentation](https://huggingface.co/datasets/openai_humaneval)
-->

## LLM-as-a-judge

While the out-of-the-box benchmarks provided by Oumi cover a broad spectrum of generative use cases, we understand that many specialized applications require more tailored evaluation objectives. If the existing benchmarks do not fully meet your needs, Oumi offers a flexible and streamlined process to create and automate evaluations, by leveraging an {doc}`LLM Judge </user_guides/judge/judge>`.

You can author your {doc}`own set of evaluation prompts </user_guides/judge/custom_prompt>` and customize the metrics to align with your specific domain or use case. By leveraging {doc}`an LLM to assess your model's outputs </user_guides/judge/custom_infer>`, you can fully automate the evaluation pipeline, producing insightful scores that truly reflect your unique criteria.

**Resources:**
- {gh}`Oumi Judge <notebooks/Oumi - Oumi Judge.ipynb>` notebook
- {gh}`Custom Judge <notebooks/Oumi - Custom Judge.ipynb>` notebook
