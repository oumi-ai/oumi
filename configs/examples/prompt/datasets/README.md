# Prompt Optimization Datasets

This directory contains example datasets for prompt optimization. These datasets can be used to test and learn the prompt optimization feature.

## Dataset Format

Oumi supports two dataset formats for prompt optimization:

### 1. Simple Format (Recommended for most use cases)

Each line in the JSONL file should contain a JSON object with `input` and `output` fields:

```jsonl
{"input": "What is the capital of France?", "output": "Paris"}
{"input": "What is 2 + 2?", "output": "4"}
```

**Fields:**
- `input` (string, required): The input prompt/question for the model
- `output` (string, required): The expected output/answer

This format is ideal for:
- Question answering
- Classification tasks
- Generation tasks
- Any single-turn interaction

### 2. Conversation Format (For multi-turn dialogues)

Each line contains a JSON object with a `messages` field following the Oumi Conversation format:

```jsonl
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
```

**Structure:**
- `messages` (array, required): List of message objects
  - Each message has:
    - `role` (string): Either "user", "assistant", or "system"
    - `content` (string): The message content

The optimizer will use the last user message as input and the last assistant message as the expected output.

## Example Datasets

### qa_train.jsonl & qa_val.jsonl
Diverse multi-task dataset covering a wide range of capabilities:

**Task Types:**
- **Sentiment Classification** - Analyze emotional tone (positive/negative/neutral)
- **Entity Extraction** - Extract names, locations from text
- **Summarization** - Condense longer text into concise summaries
- **Mathematics** - Solve equations, percentages, sequences
- **Fact vs Opinion** - Distinguish factual statements from subjective opinions
- **Translation** - English to Spanish translations
- **Trivia/Knowledge** - Geography, science, history questions
- **Classification** - Categorize animals, genres, shapes, food groups
- **Grammar** - Identify verbs, parts of speech, tense conversion
- **Reasoning** - Analogies, pattern completion, opposites
- **Measurements** - Unit conversions, temperature scales

**Training set:** 104 diverse examples (5x larger than before!)
**Validation set:** 30 diverse examples (3x larger than before!)
**Format:** Simple (input/output)

This diversity ensures that optimized prompts learn general instruction-following patterns rather than overfitting to a single task type, making them more robust and useful for real-world applications.

## Creating Your Own Dataset

### For Question Answering

1. Create a JSONL file with one example per line
2. Each line should have "input" and "output" fields
3. Keep questions clear and concise
4. Ensure answers are consistent in format

Example:
```python
import json

data = [
    {"input": "What is AI?", "output": "Artificial Intelligence"},
    {"input": "Define ML", "output": "Machine Learning"},
]

with open("my_dataset.jsonl", "w") as f:
    for example in data:
        f.write(json.dumps(example) + "\n")
```

### For Classification

For classification tasks, format as question-answer pairs:

```jsonl
{"input": "Classify sentiment: This movie was amazing!", "output": "positive"}
{"input": "Classify sentiment: Terrible experience", "output": "negative"}
```

### For Generation

Provide examples of desired output style:

```jsonl
{"input": "Summarize: Long article text here...", "output": "Brief summary"}
{"input": "Translate to French: Hello", "output": "Bonjour"}
```

## Dataset Size Recommendations

| Optimizer | Minimum Examples | Recommended Examples | Best Use Case |
|-----------|-----------------|---------------------|---------------|
| Bootstrap | 20 (enforced) | 50+ | Small datasets, few-shot selection |
| MIPRO | 20 (enforced) | 300+ | Large datasets, instruction optimization |
| GEPA | 20 (enforced) | 100+ | Complex tasks, reflective optimization |

**Note**: The system now requires a minimum of 20 training examples and 10 validation examples for meaningful optimization. This ensures reliable results and prevents wasted compute on datasets that are too small.

## Data Quality Tips

1. **Consistency**: Keep answer formats consistent across examples
2. **Diversity**: Include various types of questions/inputs
3. **Quality over Quantity**: Better to have 50 high-quality examples than 500 poor ones
4. **Balance**: For classification, balance classes in your dataset
5. **Validation Split**: Reserve 10-20% of data for validation
6. **No Duplicates**: Remove duplicate examples
7. **Clean Data**: Fix typos, formatting issues, incorrect labels

## Validation

To validate your dataset before optimization:

```python
import json
from pathlib import Path

def validate_dataset(file_path):
    with open(file_path) as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                assert "input" in data and "output" in data
                assert isinstance(data["input"], str)
                assert isinstance(data["output"], str)
                assert data["input"].strip() and data["output"].strip()
            except Exception as e:
                print(f"Error on line {i}: {e}")
                return False
    return True

if validate_dataset("my_dataset.jsonl"):
    print("Dataset is valid!")
```

## Common Issues

### Issue: "No training examples could be converted"
**Solution**: Ensure each line has both "input" and "output" fields as strings

### Issue: "Training set has only X examples"
**Solution**: Add more examples or reduce the validation split

### Issue: Empty outputs
**Solution**: Check that output fields are not empty strings

### Issue: Inconsistent results
**Solution**: Ensure answer formats are consistent (e.g., always "Yes"/"No", not "yes"/"YES"/"Y")

## Using Custom Datasets

In your configuration file:

```yaml
train_dataset_path: "path/to/your/train.jsonl"
val_dataset_path: "path/to/your/val.jsonl"  # Optional

# Or let Oumi split for you:
train_dataset_path: "path/to/your/data.jsonl"
# val_dataset_path not specified = automatic 80/20 split
```

## Next Steps

1. Choose a dataset or create your own
2. Select an appropriate optimizer for your dataset size
3. Configure your optimization in a YAML file
4. Run: `oumi prompt --config your_config.yaml`

See the parent README for more information on prompt optimization configuration.
