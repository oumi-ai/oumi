# Standardized Benchmarks

Standardized benchmarks are important for evaluating LLMs because they provide a consistent and objective way to compare the performance of different models across various tasks. This allows researchers and developers to accurately assess progress, identify strengths and weaknesses, all while ensuring fair comparisons between different LLMs.

## Overview

These benchmarks assess a model's general and domain-specific knowledge, its comprehension and ability for commonsense reasoning and logical analysis, entity recognition, factuality and truthfulness, as well as mathematical and coding capabilities. In standardized benchmarks, the prompts are structured in a way so that possible answers can be predefined.

The most common method to limit the answer space for standardized tasks is asking the model to select the correct answer from set of multiple-choice options (e.g., A, B, C, D), based on its understanding and reasoning about the input. Another way is limiting the answer space to a single word or a short phrase, which can be directly extracted from the text. In this case, the model's task is to identify the correct word/phrase that answers a question or matches the entity required. An alternative setup is asking the model to chronologically rank a set of statements, rank them to achieve logical consistency, or rank them on metrics such as plausibility/correctness, importance, or relevance. Finally, fill-in-the-blank questions, masking answer tasks, and True/False questions are also popular options for limiting the answer space.

We use EleutherAI's [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) platform as Oumi's backend to power scalable, high-performance evaluations of LLMs, providing robust and consistent benchmarking across a wide range of [standardized tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

## Trade-offs

### Advantages

The closed nature of standardized benchmarks allows for more precise and objective evaluation, focusing on a model's ability to understand, reason, and extract information accurately. The benchmarks assess a wide range of model skills in a controlled and easily quantifiable way.

1. **Objective and consistent evaluation**. With a closed answer space, there are no subjective interpretations of what constitutes the correct answer, since there’s a clear right answer among a set of predefined choices. This ensures consistency in scoring, allowing evaluators to use standard metrics (F1 score, precision, recall, accuracy, etc.) in a straightforward manner. In addition, results from different models can be directly compared because the possible answers are fixed, ensuring consistency across evaluations.

2. **Reproducibility**. When models are tested on the same benchmark with the same set of options, other researchers can replicate the results and verify claims, as long as (i) all the environmental settings are the same (Oumi thoroughly logs all settings that could affect evaluation variability) and (ii) the model is prompted with temperature 0.0 and a consistent seed. Reproducibility is crucial to track improvements across models or versions, as well as scientific rigor and advancing the state of the art in AI research.

3. **Task and domain diversity**. These benchmarks have very wide coverage and include a broad spectrum of tasks, which can highlight specific areas where a model excels or falls short. They reflect real-world challenges and complexities. There is also a multitude of benchmarks that test a model on domain-specific intricacies, assessing its ability to apply specialized knowledge within a particular field, ensuring that evaluation is closely tied to practical performance.

4. **Low cost inference and development**. In closed spaces, the model's output is often a straightforward prediction (e.g., a multiple choice letter or a single word), which is less resource-intensive since it only requires generating a few tokens (vs. a complex full-text response). In addition, the model doesn't need to consider an infinite range of possible responses, it can focus its reasoning or search on a smaller, fixed set of options, also contributing in faster inference. Developing such benchmarks also involves a simpler annotation process and low-cost labelling.

### Limitations

While standardized benchmarks offer several advantages, they also come with several limitations compared to generative benchmarks, especially in assessing the broader, more complex language abilities that are required in many real-world applications such as creativity or nuanced reasoning.

1. **Open-ended problem solving and novelty**: Models are not tested on their ability to generate creative or novel responses, explain the steps required to address a problem, being aware of the previous context to keep a conversation engaging, or to handle tasks where there isn’t a single correct answer. Many real-world applications, such as conversational agents, generating essays and stories, or summarization demand open-ended problem solving.

2. **Language quality and human alignment**. In tasks that require text generation, the style, fluency, and coherence of a model's output are crucial. Closed-answer benchmarks do not assess how well a model can generate meaningful, varied, or contextually rich language. Adapting to a persona or tone, if requested by the user, is also not assessed. Finally, alignment with human morals and social norms, being diplomatic when asked controversial questions, understanding humor and being culturally aware are outside the scope of standardized benchmarks.

3. **Ambiguity**. Closed-answer benchmarks do not evaluate the model's ability to handle ambiguous prompts. This is a common real-word scenario and an important conversational skill for agents. Addressing ambiguity typically involves asking for clarifications, requesting more context, or engaging in a dynamic context-sensitive back-and-forth conversation with targeted questions until the user's intention is revealed and becomes clear and actionable.

4. **Overfitting and cheating**. Boosting performance on standardized benchmarks requires that the model is trained on similar benchmarks. However, since the answer space is fixed and closed, models may overfit and learn to recognize patterns that are only applicable to multiple choice answers, struggling to generalize in real-world scenarios where the "correct" answer isn’t part of a predefined set. In addition, intentionally or unintentionally training on the test set is an emerging issue, which is recently (only partially) addressed by contamination IDs.

## Popular Benchmarks

### HuggingFace Leaderboard V2

As of early 2025, the most popular standardized benchmarks, used across academia and industry, are the benchmarks introduced by HuggingFace's latest (V2) leaderboard. HuggingFace has posted [a blog](https://huggingface.co/spaces/open-llm-leaderboard/blog) elaborating on why these benchmarks have been selected, while EleutherAI has also provided a comprehensive [README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md) discussing the benchmark evaluation goals, coverage, and applicability. You can find how to evaluate with these benchmarks in our {doc}`Leaderboards page </user_guides/evaluate/leaderboards>`.

- MMLU-Pro (Massive Multitask Language Understanding) [[paper](https://arxiv.org/abs/2406.01574)]
- GPQA (Google-Proof Q&A Benchmark) [[paper](https://arxiv.org/abs/2311.12022)]
- MuSR (Multistep Soft Reasoning) [[paper](https://arxiv.org/abs/2310.16049)]
- MATH (Mathematics Aptitude Test of Heuristics, Level 5). [[paper](https://arxiv.org/abs/2103.03874)]
- IFEval (Instruction Following Evaluation) [[paper](https://arxiv.org/abs/2311.07911)]
- BBH (Big Bench Hard) [[paper](https://arxiv.org/abs/2210.09261)]

### HuggingFace Leaderboard V1

Before HuggingFace's leaderboard V2 was introduced, the most popular benchmarks were captured in the [V1 leaderboard](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive). You can find how to evaluate with these benchmarks in our {doc}`Leaderboards page </user_guides/evaluate/leaderboards>`. However, due to the fast advancement of AI models, many of these benchmarks have been saturated (i.e., they became too easy to measure meaningful improvements for recent models) while newer models also showed signs of contamination, indicating that data very similar to these benchmarks may exist in their training sets.

- ARC (AI2 Reasoning Challenge) [[paper](https://arxiv.org/abs/1803.05457)]
- MMLU (Massive Multitask Language Understanding) [[paper](https://arxiv.org/abs/2009.03300)]
- Winogrande (Adversarial Winograd Schema Challenge at Scale) [[paper](https://arxiv.org/abs/1907.10641)]
- HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations) [[paper](https://arxiv.org/abs/1905.07830)]
- GSM 8K (Grade School Math) [[paper](https://arxiv.org/abs/2110.14168)]
- TruthfulQA (Measuring How Models Mimic Human Falsehoods) [[paper](https://arxiv.org/abs/2109.07958)]

### Other Benchmarks

Other benchmarks that have been very popular in the past (especially in academia) but are becoming less relevant as LLMs are advancing are listed below. These are all still available for Oumi evaluations.

| Task | Type | Description | Focus |
|------|------|-------------|-------|
BoolQ | Question Answering | BoolQ is a dataset for binary question answering, where the model is asked a yes/no question based on a passage, and it must decide whether the answer is "True" or "False." It is useful for testing a model’s ability to answer factual questions and assess whether it can reason about yes/no questions from provided contexts |  Binary question answering
TriviaQA | Question Answering | TriviaQA is a large-scale dataset for question answering that includes trivia questions with answers that are retrieved from web documents. The dataset focuses on general knowledge and requires models to answer questions using factual information from the web | General knowledge question answering, fact retrieval.
CommonsenseQA | Commonsense Reasoning | CommonsenseQA is a benchmark focused on commonsense reasoning. The task involves answering multiple-choice questions that test common knowledge about the world, requiring models to use reasoning based on everyday experiences and situations. The questions cover a wide variety of domains and situations that are difficult for models to answer without commonsense knowledge | Commonsense reasoning, multiple-choice question answering.
CoQA (Conversational Question Answering) | Conversational Question Answering | CoQA is a conversational question answering dataset that tests a model's ability to engage in multi-turn dialogue and answer questions based on a given passage of text. The task requires maintaining context across multiple questions and answers, simulating the back-and-forth nature of real conversations | Conversational question answering, multi-turn dialogue
WiC (Word-in-Context) | Word Sense Disambiguation | WiC is a benchmark for evaluating word sense disambiguation, where the task is to determine whether a word has the same meaning in two different contexts. The model is given two sentences with the same word, and it must decide if the word has the same meaning in both contexts | Word sense disambiguation and contextual understanding
DROP (Discrete Reasoning Over Paragraphs) | Question Answering / Reasoning | DROP is a benchmark designed to test a model's ability to perform discrete reasoning over paragraphs. It includes questions that require arithmetic operations, counting, and other forms of reasoning on the text. The dataset includes questions with answers that involve direct retrieval, aggregation, and reasoning over the content of paragraphs | Discrete reasoning, arithmetic, and comprehension.
NQ Open (Natural Questions Open) | Open-Domain Question Answering | NQ Open is a benchmark derived from the original Natural Questions (NQ) dataset, specifically focused on open-domain question answering. Unlike the original NQ, where the answers are typically in the text, NQ Open includes questions where the answer may not directly be present in the document, testing a model's ability to reason over text and provide a relevant, accurate response | Open-domain question answering, long-form reading comprehension.
LAMBADA | Language Modeling | LAMBADA is a benchmark designed to test a model's ability to understand and predict the final word of a passage. It consists of a dataset of passages where the last word is omitted, and the model must predict it. The tasks assess the model's understanding of context, coherence, and long-range dependencies | Contextual understanding and word prediction
PIQA | Physical Reasoning | The Physical Interaction Question Answering (PIQA) benchmark tests a model’s ability to reason about physical interactions and solve problems that require practical knowledge of how the world works | Commonsense physical reasoning and understanding of everyday interactions
RTE (Recognizing Textual Entailment) | Natural Language Inference | RTE is a widely used benchmark for natural language inference (NLI), where the task is to determine whether a premise entails, contradicts, or is neutral towards a hypothesis. RTE involves diverse linguistic phenomena and tests a model's ability to understand relationships between sentences | Textual entailment and logical reasoning
SIQA (Social-IQ) | Social Reasoning | SIQA is a dataset designed to evaluate social commonsense reasoning. It contains questions based on short scenarios that require understanding of social situations and behaviors (e.g., interpreting emotions, actions, or social dynamics). The task aims to assess how well models can reason about human social interactions | Social commonsense reasoning, emotion and behavior interpretation
WMT 2016 | Machine Translation | WMT 2016 is part of the Workshop on Statistical Machine Translation (WMT) shared task, focusing on machine translation. It includes a set of parallel corpora for translation between English and various other languages, evaluating translation quality using metrics such as BLEU. WMT 2016 is one of the widely used benchmarks in the field of machine translation | Machine translation, cross-lingual language generation
SWAG | Commonsense Reasoning / Story Completion | SWAG is a benchmark for commonsense reasoning, specifically focused on story completion. The task presents a short story and asks the model to select the most plausible continuation from a set of options | Commonsense reasoning in narrative contexts
SQuAD (Stanford Question Answering Dataset) | Reading Comprehension | SQuAD is one of the most well-known benchmarks for evaluating a model's ability to answer questions based on a given passage of text. In SQuAD v1.1, questions are fact-based and answers are directly extracted from the text. SQuAD v2.0 also includes unanswerable questions to test a model's ability to determine when no answer is available | Question answering and reading comprehension
LAMA (LAnguage Model Analysis) | Knowledge Probe | LAMA is a benchmark that evaluates a model’s ability to retrieve factual knowledge. It involves filling in blanks in a sentence (e.g., “The capital of France is __”) and testing whether the model can correctly answer based on its stored knowledge | Knowledge retrieval and fact-based reasoning
ANLI (Adversarial NLI) | Natural Language Inference | ANLI is an adversarially constructed benchmark for natural language inference (NLI), designed to test models' ability to understand logical relationships between sentences. It’s a challenging task where models must determine whether a premise entails, contradicts, or is neutral toward a given hypothesis | Logical reasoning and sentence relationship understanding
RACE (ReAding Comprehension from Examinations) | Reading Comprehension | RACE is a reading comprehension benchmark that contains a large number of English exam questions drawn from Chinese middle and high school exams. It’s designed to test the ability of LLMs to handle long, complex passages of text | Long-form reading comprehension
CLOTH (Commonsense Reasoning in Texts) | Textual Commonsense Reasoning | CLOTH is a benchmark for commonsense reasoning involving textual entailment. It includes tasks that require a deep understanding of everyday situations and textual inference | Commonsense reasoning, textual entailment


To see all available tasks:

```bash
lm-eval --tasks list
```

## Custom LM-Harness Tasks

While Oumi provides integration with the LM Evaluation Harness and its extensive task collection, you may need to create a custom evaluation tasks for specific use cases. For this case, we refer you to the [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md), which walks you through the process of creating and implementing custom evaluation tasks using the `LM Evaluation Harness` (`lm_eval`) framework.
