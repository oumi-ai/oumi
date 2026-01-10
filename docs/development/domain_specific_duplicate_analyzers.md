# Domain-Specific Duplicate Analyzers

## Overview

Domain-specific duplicate analyzers provide targeted duplication detection with appropriate thresholds for different types of conversation content. Unlike the generic `duplicate` analyzer that treats all content equally, these analyzers understand that different message types have different acceptable duplication levels.

## Why Domain-Specific Analyzers?

The investigation of the Alpaca dataset revealed that **34.4% duplicate rate was misleading** because:
- 97% of "duplicates" were actually system prompt templates (expected and normal)
- Only ~1% were actual content duplicates

**Key insight**: Different roles have different duplication expectations:
- **System prompts**: 80-100% duplication is normal and expected
- **Questions (user messages)**: 5-15% duplication is acceptable
- **Responses (assistant messages)**: <5% duplication expected
- **QA pairs**: Should be mostly unique

## Available Analyzers

### 1. System Prompt Analyzer (`system_prompt`)

**Purpose**: Analyzes system role messages for templates, patterns, and consistency.

**Use Case**: Datasets with system prompts (instruction templates, persona definitions, formatting instructions)

**Metrics**:
- `system_prompt_missing`: Whether conversation lacks a system prompt
- `system_prompt_hash`: Hash of system prompt content
- `system_prompt_is_common_template`: Whether prompt matches common template (>= min_template_frequency)
- `system_prompt_template_rank`: Rank by frequency (1 = most common)
- `system_prompt_is_unusual`: Whether prompt is rare/unique
- `system_prompt_length`: Character count
- `system_prompt_duplicate_count`: Number of occurrences

**Parameters**:
```yaml
- id: system_prompt
  params:
    expected_duplication_threshold: 0.80  # 80%+ duplication is normal
    max_unique_templates: 10              # Max expected unique templates
    min_template_frequency: 0.05          # Templates below 5% frequency flagged as unusual
    normalize_whitespace: true
    case_sensitive: false
```

**Thresholds**:
- **Normal**: 80-100% of samples share the same system prompt
- **Concern**: >10 unique system prompts (suggests inconsistent formatting)
- **Flag**: Templates appearing in <5% of samples (outliers)

**Example Findings**:
```
✅ Normal: "You are a helpful assistant" appears in 95% of conversations
⚠️  Unusual: "You are a Python expert" appears in only 2% of conversations
❌ Issue: 50 different system prompts found (suggests inconsistent dataset preparation)
```

---

### 2. Question-Answer Pair Analyzer (`qa_pair_duplicate`)

**Purpose**: Detects duplicate (question, answer) combinations.

**Use Case**: QA datasets, instruction-following datasets, dialogue datasets

**Metrics**:
- `qa_pair_hash`: Hash of (question, answer) pair
- `qa_pair_duplicate_count`: Number of times this exact pair appears
- `qa_pair_is_duplicate`: Whether pair is duplicated (>1 occurrence)
- `qa_pair_high_duplication`: Whether duplication exceeds threshold

**Parameters**:
```yaml
- id: qa_pair_duplicate
  params:
    duplicate_threshold: 0.05  # >5% QA pair duplication is concerning
    normalize_whitespace: true
    case_sensitive: false
```

**Thresholds**:
- **Normal**: <5% duplicate QA pairs
- **Concern**: 5-10% duplicate pairs
- **Issue**: >10% duplicate pairs (indicates redundant training data)

**Example Findings**:
```
✅ Good: 52,000 samples, 51,800 unique QA pairs (0.4% duplication)
⚠️  Moderate: 52,000 samples, 48,000 unique QA pairs (7.7% duplication)
❌ High: 52,000 samples, 40,000 unique QA pairs (23% duplication - remove duplicates!)
```

---

### 3. Question Duplicate Analyzer (`question_duplicate`)

**Purpose**: Detects duplicate user questions/instructions.

**Use Case**: Understanding question diversity in training data

**Metrics**:
- `question_hash`: Hash of question content
- `question_duplicate_count`: Number of times this question appears
- `question_is_duplicate`: Whether question is duplicated (>1 occurrence)
- `question_duplication_level`: Classification ("unique", "acceptable", "moderate", "high")

**Parameters**:
```yaml
- id: question_duplicate
  params:
    acceptable_duplication: 0.15          # 15% question duplication is acceptable
    high_duplication_threshold: 0.20      # >20% is concerning
    normalize_whitespace: true
    case_sensitive: false
```

**Thresholds**:
- **Normal**: <15% duplicate questions (multiple users ask the same thing)
- **Moderate**: 15-20% duplicate questions
- **High**: >20% duplicate questions (lack of diversity)

**Duplication Levels**:
- `unique`: Question appears once
- `acceptable`: Duplicate but below 15% threshold
- `moderate`: 15-20% duplication rate
- `high`: >20% duplication rate

**Example Findings**:
```
✅ Good: "What is Python?" appears 10 times in 10,000 samples (0.1% per question)
⚠️  Moderate: "How do I...?" variations appear in 18% of samples
❌ High: Top 5 questions account for 25% of all samples (need more diversity)
```

---

### 4. Response Duplicate Analyzer (`response_duplicate`)

**Purpose**: Detects duplicate assistant responses.

**Use Case**: Identifying generic/templated responses, ensuring response diversity

**Metrics**:
- `response_hash`: Hash of response content
- `response_duplicate_count`: Number of times this response appears
- `response_is_duplicate`: Whether response is duplicated (>1 occurrence)
- `response_is_short`: Whether response is short (<= short_response_length chars)
- `response_duplication_level`: Classification ("unique", "acceptable", "acceptable_short", "moderate", "high")
- `response_is_generic`: Whether response is very common (>1% of all responses)

**Parameters**:
```yaml
- id: response_duplicate
  params:
    acceptable_duplication: 0.05          # 5% response duplication is acceptable
    high_duplication_threshold: 0.10      # >10% is concerning
    short_response_length: 20             # Responses <20 chars have higher expected duplication
    normalize_whitespace: true
    case_sensitive: false
```

**Thresholds**:
- **Normal**: <5% duplicate responses
- **Moderate**: 5-10% duplicate responses
- **High**: >10% duplicate responses (too generic)
- **Generic**: Response appears in >1% of all samples (very common)

**Duplication Levels**:
- `unique`: Response appears once
- `acceptable`: Duplicate but below 5% threshold
- `acceptable_short`: Short response (<20 chars) with duplication (expected)
- `moderate`: 5-10% duplication rate
- `high`: >10% duplication rate

**Example Findings**:
```
✅ Good: 52,000 responses, 51,500 unique (0.96% duplication)
⚠️  Short responses: "Yes" appears 200 times (0.4% - acceptable for short response)
⚠️  Generic: "I don't know" appears 800 times (1.5% - very common generic response)
❌ High: "Here is the answer:" appears 6,000 times (11.5% - templated responses!)
```

---

## Requirements

All domain-specific analyzers require datasets with explicit **role** columns:
- `role`: Message role (system, user, assistant)
- `text_content`: Message content
- `conversation_id`: Conversation identifier (optional but recommended)
- `message_index`: Message position in conversation (optional but recommended)

**Supported dataset formats**:
- ✅ Oumi conversation format (with role column)
- ✅ ShareGPT format (with role annotations)
- ✅ Any format with explicit role column
- ❌ Alpaca format (instruction/input/output) - requires conversion
- ❌ Simple text datasets without roles

## Usage

### Basic Configuration

```yaml
# configs/examples/analyze/domain_specific_dedup.yaml
dataset_name: your-dataset/name
split: train
sample_count: null  # Analyze all

output_path: ./dedup_analysis

tokenizer_name: openai-community/gpt2

analyzers:
  # System prompt analysis
  - id: system_prompt
    params:
      expected_duplication_threshold: 0.80
      max_unique_templates: 10

  # QA pair deduplication
  - id: qa_pair_duplicate
    params:
      duplicate_threshold: 0.05

  # Question diversity
  - id: question_duplicate
    params:
      acceptable_duplication: 0.15
      high_duplication_threshold: 0.20

  # Response diversity
  - id: response_duplicate
    params:
      acceptable_duplication: 0.05
      high_duplication_threshold: 0.10
      short_response_length: 20
```

### Run Analysis

```bash
oumi analyze --config configs/examples/analyze/domain_specific_dedup.yaml
```

### Interpret Results

Check the analysis summary:
```bash
cat ./dedup_analysis/analysis_summary.json
```

Filter for issues:
```bash
# Find conversations with missing system prompts
csvgrep -c system_prompt_missing -m True ./dedup_analysis/conversation_analysis.csv

# Find duplicate QA pairs
csvgrep -c qa_pair_is_duplicate -m True ./dedup_analysis/message_analysis.csv

# Find high duplication questions
csvgrep -c question_duplication_level -m high ./dedup_analysis/message_analysis.csv

# Find generic responses
csvgrep -c response_is_generic -m True ./dedup_analysis/message_analysis.csv
```

---

## Recommendations by Dataset Type

### Instruction-Following Datasets (e.g., Alpaca, FLAN)

**Use**:
- `system_prompt`: Check consistency of instruction templates
- `question_duplicate`: Ensure instruction diversity (15-20% duplication OK)
- `response_duplicate`: Flag generic responses (<5% duplication target)

**Skip**:
- `qa_pair_duplicate`: May not apply if instructions intentionally paired with varied responses

### Dialogue/Chat Datasets (e.g., ShareGPT, OpenAssistant)

**Use**:
- `system_prompt`: Check persona/role consistency
- `qa_pair_duplicate`: Remove duplicate conversations (<5% target)
- `question_duplicate`: Monitor natural question diversity (15-20% OK)
- `response_duplicate`: Ensure varied, contextual responses (<5% target)

### QA Datasets (e.g., SQuAD, Natural Questions)

**Use**:
- `qa_pair_duplicate`: **Critical** - exact duplicates should be removed (<1% target)
- `question_duplicate`: Some duplication acceptable (10-15%)
- `response_duplicate`: Low duplication expected (<5%)

**Skip**:
- `system_prompt`: May not have system prompts

---

## Comparison with Generic Duplicate Analyzer

| Aspect | Generic `duplicate` | Domain-Specific Analyzers |
|--------|-------------------|--------------------------|
| **Granularity** | All content treated equally | Role-specific analysis |
| **Thresholds** | One threshold for all | Appropriate per role |
| **Context** | No role awareness | Understands conversation structure |
| **System Prompts** | Flags as duplicates (wrong) | Expects duplication (correct) |
| **Short Responses** | Flags "Yes"/"No" duplicates | Accepts short response duplication |
| **QA Pairs** | Can't detect | Detects (Q,A) combination duplicates |
| **Use Case** | Quick check | Production data quality |

**Recommendation**: Use **both**:
1. Generic `duplicate` for initial broad detection
2. Domain-specific analyzers for actionable insights

---

## Examples

### Example 1: Clean Dataset

```yaml
System Prompts:
  ✅ 95% share "You are a helpful assistant" template
  ✅ 2 unique templates total
  ✅ No missing system prompts

QA Pairs:
  ✅ 0.4% duplicate pairs (201 / 52,000)
  ✅ Well below 5% threshold

Questions:
  ✅ 8% question duplication
  ✅ Classification: "acceptable"

Responses:
  ✅ 1.2% response duplication
  ✅ Generic responses: 3 (0.006%)
  ✅ Classification: "acceptable"

**Action**: ✅ Dataset ready for training
```

### Example 2: Dataset Needs Cleaning

```yaml
System Prompts:
  ⚠️  15 unique templates found
  ❌ 23% missing system prompts
  ⚠️  5 templates appear <5% frequency

QA Pairs:
  ❌ 15% duplicate pairs (7,800 / 52,000)
  ❌ Exceeds 5% threshold

Questions:
  ⚠️  22% question duplication
  ⚠️  Classification: "high"

Responses:
  ❌ 12% response duplication
  ❌ "I'm sorry, I can't help with that" appears 3,200 times (6%)
  ❌ Classification: "high"

**Action**:
1. Remove 7,800 duplicate QA pairs
2. Standardize system prompts
3. Review and vary generic responses
4. Consider adding more diverse questions
```

---

## Testing

Run tests:
```bash
pytest tests/unit/core/analyze/test_domain_specific_analyzers.py -v
```

Test coverage:
- SystemPromptAnalyzer: 3 tests
- QuestionAnswerPairAnalyzer: 2 tests
- QuestionDuplicateAnalyzer: 3 tests
- ResponseDuplicateAnalyzer: 5 tests

**Total**: 13 tests, all passing ✅

---

## Future Enhancements

1. **Multi-turn context awareness**: Detect duplicates in conversation sequences
2. **Semantic deduplication**: Use embeddings to find near-duplicates
3. **Language-specific handling**: Different thresholds for different languages
4. **Automated cleaning**: Generate deduplication recommendations
5. **Integration with training**: Filter duplicates during data loading

---

## See Also

- [Dataset Quality Roadmap](dataset_quality_roadmap.md)
- [Analyzer Accuracy Report](/tmp/analyzer_accuracy_report.md)
- [Analysis Config Examples](../../configs/examples/analyze/)
