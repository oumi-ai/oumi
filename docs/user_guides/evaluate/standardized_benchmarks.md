# Standardized Benchmarks

Standardized benchmarks are important for evaluating LLMs because they provide a consistent and objective way to compare the performance of different models across various tasks. This allows researchers and developers to accurately assess progress, identify strengths and weaknesses, all while ensuring fair comparisons between different LLMs.

## Overview

These benchmarks assess a model's general and domain-specific knowledge, its comprehension and ability for commonsense reasoning and logical analysis, entity recognition, factuality and truthfulness, as well as mathematical and coding capabilities. In standardized benchmarks, the prompts are structured in a way so that possible answers can be predefined.

The most common method to limit the answer space for standardized tasks is asking the model to select the correct answer from set of multiple-choice options (e.g., A, B, C, D), based on its understanding and reasoning about the input. Another way is limiting the answer space to a single word or a short phrase, which can be directly extracted from the text. In this case, the model's task is to identify the correct word/phrase that answers a question or matches the entity required. An alternative setup is asking the model to chronologically rank a set of statements, rank them to achieve logical consistency, or rank them on metrics such as plausibility/correctness, importance, or relevance. Finally, fill-in-the-blank questions, masking answer tasks, and True/False questions are also popular options for limiting the answer space.

Oumi uses EleutherAI’s [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to power scalable, high-performance evaluations of LLMs, providing robust and consistent benchmarking across a wide range of [standardized tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

## Popular Benchmarks

This section discusses the most popular standardized benchmarks, in order to give you a starting point for your evaluations. You can kick-off evaluations using the following configuration template. For advanced configuration settings, please visit the {doc}`evaluation configuration </user_guides/evaluate/evaluation_config>` page.

```yaml
model:
  model_name: <HuggingFace model name or local path to model>
  trust_remote_code: False # Set to true for HuggingFace models

tasks:
  - evaluation_backend: lm_harness
    task_name: <`Task Name` from the tables below>
    eval_kwargs:
      num_fewshot: <number of few-shot prompts, if applicable>

output_dir: <output directory>
```

To see all supported standardized benchmarks:

```bash
lm-eval --tasks list
```

### Question Answering and Knowledge Retrieval
Benchmarks that evaluate a model's ability to understand questions and generate accurate answers, based on the provided context (Open-Book) or its internal knowledge (Closed-Book).

| Task | Description | Type | Task Name | Introduced |
|------|-------------|------|-----------|------------|
BoolQ (Boolean Questions) | A question-answering task consisting of a short passage from a Wikipedia article and a yes/no question about the passage [[details](https://arxiv.org/abs/1905.00537)] | Open-Book (True/False answer) | `boolq` | 2019, as part of Superglue
TriviaQA | Trivia question answering to test general knowledge, using evidence documents [[details](https://nlp.cs.washington.edu/triviaqa/)] | Open-Book (free-form answer) | `triviaqa` | 2017, by UW in ACL
CoQA (Conversational Question Answering) | Measure the ability of machines to understand a text passage and answer a series of interconnected questions that appear in a conversation [[details](https://arxiv.org/abs/1808.07042)] | Open-Book (free-form answer) | `coqa` | 2018, by Stanford in TACL
NQ (Natural Questions) | Open domain question answering benchmark that is derived from Natural Questions. The goal is to predict an answer for a  question in English [[details](https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/)] | Closed-Book (free-form answer) | `nq_open` | 2019, by Google in TACL
SQuAD V2 (Stanford Question Answering Dataset) | Reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles. The answer is either a segment of text from the reading passage or unanswerable [[details](https://arxiv.org/abs/1806.03822)] | Open-Book (free-form answer) | `squadv2` | 2018, by Stanford in ACL
GPQA (Google-Proof Q&A) | Very difficult multiple-choice questions written by domain experts in biology, physics, and chemistry [[details](https://arxiv.org/abs/2311.12022)] | Closed-Book (multichoice answer) | `gpqa` | 2023, by NYU, Cohere, Anthropic
ARC Challenge (AI2 Reasoning Challenge) | Challenging multiple-choice science questions from the ARC dataset. Answered incorrectly by standard retrieval-based and word co-occurrence algorithms [[details](https://arxiv.org/abs/1803.05457)] | Closed-Book (multichoice answer) | `arc_challenge` | 2018, by Allen AI
MMLU (Massive Multitask Language Understanding) | Multiple choice QA benchmark on elementary mathematics, US history, computer science, law, and more [[details](https://arxiv.org/abs/2009.03300)] | Closed-Book (multichoice answer) | `mmlu` | 2021, by Berkeley, Columbia and others in ICLR
MMLU Pro (Massive Multitask Language Understanding) | Enhanced MMLU extending the knowledge-driven questions by integrating more challenging, reasoning-focused questions and expanding the choice set from four to ten options [[details](https://arxiv.org/abs/2406.01574)] | Closed-Book (multichoice answer) | `mmlu_pro` | 2024, by U Waterloo, U Toronto, CMU
Truthful QA | Measures if model mimics human falsehoods. Assesses truthfulness and ability to avoid humans' false beliefs or misconceptions (38 categories, including health, law, finance and politics) [[details](https://arxiv.org/abs/2109.07958)] | Open-Book (both multichoice and free-form) | `truthfulqa_mc2` | 2022, by University of Oxford, OpenAI

### Commonsense and Logical Reasoning
Benchmarks that assess a model's ability to perform reasoning tasks requiring commonsense understanding and logical thinking.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
Commonsense QA | Multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answer among one correct answer and four distracting answers [[details](https://arxiv.org/pdf/1811.00937.pdf)] | `commonsense_qa` | 2019, by Tel-Aviv University and Allen AI
PIQA (Physical Interaction QA) | Physical commonsense reasoning to investigate the physical knowledge of existing models, including basic properties of the real-world objects  [[details](https://arxiv.org/abs/1911.11641)] | `piqa` | 2019, by Allen AI, MSR, CMU, UW
SocialIQA (Social Interaction QA) | Commonsense reasoning about social situations, probing emotional and social intelligence in a variety of everyday situations [[details](https://arxiv.org/abs/1904.09728)] | `siqa` | 2019, by Allen AI, UW
SWAG (Situations With Adversarial Generations) | Grounded commonsense reasoning. Questions sourced from video captions with answers being what might happen next in the next scene (1 correct and 3 adversarially generated choices) [[details](https://arxiv.org/abs/1808.05326)] | `swag` | 2019, by UW, Allen AI
HellaSWAG | Benchmark that builds on SWAG to evaluate understanding and common sense reasoning, particularly in the context of completing sentences or narratives [[details](https://arxiv.org/abs/1905.07830)] | `hellaswag` | 2019, by UW, Allen AI
WinoGrande | Given a sentence which requires commonsense reasoning, choose the right option among multiple choices. Inspired by Winograd Schema Challenge (WSC) [[details](https://arxiv.org/abs/1907.10641)] | `winogrande` | 2019, by Allen AI
MuSR (Multistep Soft Reasoning) | Multistep soft reasoning tasks specified in a natural language narrative. Includes solving murder mysteries, object placement, and team allocation [[details](https://arxiv.org/abs/2310.16049)] | `leaderboard_musr` | 2024, by UT Austin in ICLR
DROP (Discrete Reasoning Over Paragraphs) | Reading comprehension benchmark. Requires reference resolution and performing discrete operations over the references (addition, counting, or sorting) [[details](https://arxiv.org/abs/1903.00161)] | `drop` | 2019, by UC Irvine, Peking University and others
ANLI (Adversarial NLI) | Reasoning dataset. Given a premise, identify if a hypothesis is entailment, neutral, or contradictory [[details](https://arxiv.org/abs/1910.14599)] | `anli` | 2019, by UNC Chapel Hill and Meta
BBH (Big Bench Hard) | Challenging tasks from the BIG-Bench evaluation suite, focusing on complex reasoning, multi-step problem solving, and requiring deep document understanding rather than surface-level pattern matching [[details](https://arxiv.org/abs/2210.09261)] | `bbh` | 2022, by Google Research and Stanford

### Language Understanding
Benchmarks that test a model's understanding of language semantics and syntax.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
WiC (Words in Context) | Word sense disambiguation task. Requires identifying if occurrences of a word in two contexts correspond to the same meaning or not. Framed as a binary classification task [[details](https://arxiv.org/abs/1905.00537)] | `wic` | 2019, as part of Superglue
RTE (Recognizing Textual Entailment) | Given two text fragments, recognize whether the meaning of one fragment can be inferred from the other [[details](https://arxiv.org/abs/1905.00537)] | `rte` | 2019, as part of Superglue
LAMBADA (LAnguage Modeling Broadened to Account for Discourse Aspects) | Word prediction task. Given a passage, predict the last word. Requires tracking information in the broader discourse, beyond the last sentence [[details](https://arxiv.org/abs/1606.06031)] | `lambada` | 2016, by CIMeC, University of Trento
WMT 2016 (Workshop on Machine Translation) | Collection of parallel text data used to assess the performance of machine translation systems, primarily focusing on news articles, across various language pairs [[details](http://www.aclweb.org/anthology/W/W16/W16-2301)] | `wmt16` | 2016, by Charles University, FBK, and others
RACE (ReAding Comprehension from Examinations) | Reading comprehension dataset collected from English examinations in China. Designed for middle school and high school students. Evaluates language understanding and reasoning [[details](https://arxiv.org/abs/1704.04683)] | `race` | 2017, by CMU
IFEval (Instruction Following Evaluation) | Instruction-Following evaluation dataset. Focuses on formatting text, including imposing length constraints, paragraph composition, punctuation, enforcing lower/upper casing, including/exluding keywords, etc [[details](https://arxiv.org/abs/2311.07911)] | `ifeval` | 2023, by Google, Yale University
<!-- FIXME: Move IFEval to generative Benchmarks-->

### Mathematical and Numerical Reasoning
Benchmarks focused on evaluating a model's ability to perform mathematical calculations and reason about numerical information.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
MATH (Mathematics Aptitude Test of Heuristics), Level 5  | Challenging competition mathematics problems that require step-by-step solutions [[details](https://arxiv.org/abs/2103.03874)] | `leaderboard_math_hard` | 2021, by UC Berkeley
GSM 8K (Grade School Math) | Grade school-level math word problems [[details](https://arxiv.org/abs/2110.14168)] | `gsm8k` | 2021, by OpenAi

(multi-modal-standardized-benchmarks)=
### Multi-modal Benchmarks

Benchmarks to evaluate vision-language (image + text) models.

| Task | Description | Task Name | Introduced |
|------|-------------|-----------|------------|
MMMU (Massive Multi-discipline Multimodal Understanding) | Designed to evaluate multimodal (image + text) models on multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning. MMMU includes 11.5K meticulously collected multimodal questions from college exams, quizzes, and textbooks, covering six core disciplines [[details](https://arxiv.org/abs/2311.16502)] | `mmmu_val` | 2023, by OSU and others

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

<!-- suggesting to DROP this until we fully support it; currently it hurts more than helps IMO:

## Custom LM-Harness Tasks

While Oumi provides integration with the LM Evaluation Harness and its extensive task collection, you may need to create a custom evaluation tasks for specific use cases. For this case, we refer you to the [new task guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md), which walks you through the process of creating and implementing custom evaluation tasks using the `LM Evaluation Harness` (`lm_eval`) framework.

-->
