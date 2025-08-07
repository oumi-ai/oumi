# Built-In Judges

Oumi provides a comprehensive set of built-in judges that can be used out-of-the-box for common evaluation tasks. These judges are carefully designed and tested to provide reliable assessments across different domains and use cases. All built-in judges are configured to use GPT-4 by default, but you can easily change the underlying model to any hosted or local model with a [config parameter override](/user_guides/judge/judge_config.md#parameter-override).

## Generic Judges

These judges can be applied to a wide variety of use cases, since their input fields are generic: `request` (user's request), `response` (model's output).
They provide a boolean judgment (Yes/No), together with an explanation.

They can be invoked in the CLI, as follows:
```bash
oumi judge dataset \
  --config generic/<judge name> \
  --input dataset.jsonl \
  --output output.jsonl
```

| Judge Name | Description |
|------------|-------------|
| `instruction_following` | Evaluates whether a response strictly follows the instructions provided in the user's request. |
| `truthfulness` | Determines whether a response is factually accurate, grounded in verifiable information, and free from hallucinations or fabrications.|
| `topic_adherence` | Assesses whether a response stays on topic and addresses the core subject matter of the request. |
| `format_compliance` | Evaluates whether a response follows specified formatting requirements or structural guidelines. |
| `safety` | Evaluates whether a response produces or encourages harmful behavior, incites violence, promotes illegal activities, or violates ethical or social norms. |


## Document Q&A Judges

These judges are specialized for document-based question-answering scenarios. Specifically, a `context` (context document with background information) is provided and a model is asked a `question` by the user and responds with an `answer`. The answer must be retrieved from the context document only, without leveraging external sources or making assumptions.
These judges provide a boolean judgment (Yes/No), together with an explanation.

They can be invoked as follows
```bash
oumi judge dataset \
  --config doc_qa/<judge name> \
  --input dataset.jsonl \
  --output output.jsonl
```

| Judge Name | Description |
|------------|-------------|
| `relevance` | Evaluates whether an answer is relevant to a question, based on provided context. |
| `groundedness` | Determines whether an answer is properly grounded in the provided context without introducing external information. |
| `completeness` | Assesses whether an answer comprehensively addresses all aspects of the question. |
