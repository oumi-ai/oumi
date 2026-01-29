# Typed Analyzer Framework - Quick Runbook

This guide helps you get started with the **Typed Analyzer Framework**, a Pydantic-based system for analyzing datasets with strongly-typed results, custom metrics, and validation tests.

## Installation

### 1. Install Oumi with the analyze extras

```bash
# Clone the repository
git clone https://github.com/oumi-ai/oumi.git
cd oumi

# Checkout the typed analyzer branch
git checkout ryan-arman/typed-analyzer-framework

# Install with analyze dependencies
pip install -e ".[analyze]"
```

The `analyze` extras include:
- `sentence-transformers` - Semantic analysis
- `plotly` - Interactive charts
- `scikit-learn` - Clustering
- `tiktoken` - GPT-style token counting
- `datasketch` - Fuzzy duplicate detection
- `fast-langdetect` - Language detection

### 2. Optional: Install for LLM-based analyzers

If you want to use LLM-as-judge analyzers (usefulness, safety, coherence, etc.):

```bash
# Set your API key
export OPENAI_API_KEY=your-api-key

# Or for Anthropic
export ANTHROPIC_API_KEY=your-api-key
```

### 3. Optional: Install rich for better CLI output

```bash
pip install rich
```

## Quick Start

### Run with CLI

```bash
# Quick start - runs length + quality analyzers (no API key needed)
oumi analyze --config configs/examples/analyze/analyze_quick_start.yaml --typed

# Length-only analysis with advanced examples
oumi analyze --config configs/examples/analyze/typed_length_example.yaml --typed

# LLM-based analysis (requires API key)
oumi analyze --config configs/examples/analyze/typed_llm_analyzer_example.yaml --typed

# List available metrics
oumi analyze --config configs/examples/analyze/analyze_quick_start.yaml --typed --list-metrics
```

### Run with Python API

```python
from oumi.analyze import run_from_config_file, print_summary

# Run analysis
results = run_from_config_file('configs/examples/analyze/analyze_quick_start.yaml')

# Print summary
print_summary(results)

# Access results
df = results['dataframe']  # Pandas DataFrame
test_summary = results['test_summary']  # Test results (if configured)
```

### Web UI Viewer

Launch an interactive web interface to view, create, and manage analyses.

**First-time setup** (requires Node.js 18+):

```bash
cd src/oumi/analyze/web
npm install
npm run build
```

Then run:

```bash
oumi analyze view
```

This opens a browser at `http://localhost:8765` with:

**Features:**
- **Dashboard** - View all past analyses with pass/fail status
- **Results View** - Interactive table showing test results, metrics, and sample data
- **Charts View** - Visualizations of metric distributions
- **Config Editor** - View and edit YAML configurations with syntax highlighting
- **Create Analysis Wizard** - Step-by-step UI to create new analyses:
  - Select dataset (local file path or HuggingFace dataset)
  - Choose analyzers and configure parameters
  - Define validation tests
  - Run analysis directly from the UI

**Creating a New Analysis in the UI:**
1. Click **"Create New Analysis"** button
2. **Dataset step**: Enter a local file path OR a HuggingFace dataset name with split (e.g., `HuggingFaceH4/ultrachat_200k` with split `train_sft`)
3. **Analyzers step**: Select analyzers like `length`, `usefulness`, `safety`, etc.
4. **Tests step**: Add validation tests for your metrics
5. **Review step**: Preview the generated YAML config and run the analysis

**Tips:**
- For local files, type the full absolute path (e.g., `/Users/you/data/dataset.jsonl`)
- For HuggingFace datasets, check the dataset page for available splits (not always `train`)
- Results are saved and persist between sessions
- Click on any analysis in the sidebar to view its results

## Example Configurations

### 1. Quick Start (Recommended - No LLM required)

Located at: `configs/examples/analyze/analyze_quick_start.yaml`

This is the recommended starting point. It runs both the **length** and **quality** analyzers for comprehensive data validation without any API calls.

```yaml
dataset_name: HuggingFaceH4/ultrachat_200k
split: train_sft
sample_count: 50

output_path: ./analysis_output/quickstart

analyzers:
  - id: length
    params:
      tiktoken_encoding: cl100k_base
      compute_role_stats: true

  - id: quality  # Fast data quality checks
    params:
      check_turn_pattern: true      # Alternating user-assistant turns
      check_empty_content: true     # Empty messages
      check_invalid_values: true    # NaN, null, None as strings
      check_truncation: true        # Abruptly cut-off conversations
      check_refusals: true          # Policy refusal patterns
      check_tags: true              # Unbalanced think/code tags

tests:
  - id: alternating_turns
    type: percentage
    metric: quality.has_alternating_turns
    condition: "== True"
    min_percentage: 95.0
    severity: high
    title: "Proper turn structure"
```

### 2. Length Analysis (Advanced examples)

Located at: `configs/examples/analyze/typed_length_example.yaml`

```yaml
dataset_path: "/path/to/your/dataset.jsonl"
sample_count: 100

output_path: ./analysis_output/length_test

analyzers:
  - id: length
    params:
      tiktoken_encoding: cl100k_base  # GPT-4 tokenizer
      compute_role_stats: true

tests:
  - id: max_tokens
    type: threshold
    metric: length.total_tokens
    operator: ">"
    value: 8000
    max_percentage: 2.0
    severity: high
    title: "Conversations exceeding token limit"
```

### 3. LLM-based Analysis (Requires API key)

Located at: `configs/examples/analyze/typed_llm_analyzer_example.yaml`

```yaml
dataset_path: "/path/to/your/dataset.jsonl"
sample_count: 5  # Keep small due to API costs

analyzers:
  # Preset analyzers
  - id: usefulness
    params:
      model_name: gpt-4o-mini
      api_provider: openai

  - id: safety
    params:
      model_name: gpt-4o-mini
      target_scope: last_turn  # Only evaluate last exchange

  # Custom LLM criteria
  - id: llm
    params:
      criteria_name: "prompt_difficulty"
      target_scope: first_user
      judgment_type: enum
      enum_values: ["easy", "medium", "hard", "expert"]
      prompt_template: |
        Evaluate the DIFFICULTY of this user prompt...
      model_name: gpt-4o-mini
```

## Available Analyzers

| Analyzer ID | Type | Description |
|-------------|------|-------------|
| `length` | Non-LLM | Token/word counts, message statistics |
| `quality` | Non-LLM | Data quality checks: turn patterns, empty messages, truncation, refusals, tag balance |
| `usefulness` | LLM | Evaluates response usefulness |
| `safety` | LLM | Checks for unsafe content |
| `coherence` | LLM | Evaluates conversation coherence |
| `factuality` | LLM | Checks factual accuracy |
| `instruction_following` | LLM | Evaluates instruction adherence |
| `llm` | LLM | Generic LLM analyzer with custom prompts |

## Target Scopes (LLM Analyzers)

| Scope | Description |
|-------|-------------|
| `conversation` | Full multi-turn conversation |
| `last_turn` | Last user + assistant exchange |
| `system` | System prompt only |
| `first_user` | First user message |
| `last_assistant` | Last assistant response |
| `last_user` | Last user message |
| `role:user` | All user messages |
| `role:assistant` | All assistant messages |

## Custom Metrics

Define inline Python functions that compute additional metrics:

```yaml
custom_metrics:
  - id: estimated_cost
    scope: conversation
    description: "Estimated cost based on token count"
    depends_on:
      - length  # Can depend on other analyzers
    output_schema:
      - name: cost_usd
        type: float
    function: |
      def compute(conversation, results, index):
          length_result = results["length"][index]
          tokens = getattr(length_result, "total_tokens", 0) or 0
          return {"cost_usd": tokens * 0.001 / 1000}
```

## Validation Tests

Define tests that validate your metrics:

```yaml
tests:
  # Threshold test - check max percentage exceeding value
  - id: max_tokens
    type: threshold
    metric: length.total_tokens
    operator: ">"
    value: 8000
    max_percentage: 2.0
    severity: high
    title: "Token limit exceeded"

  # Percentage test - check min percentage meeting condition
  - id: high_usefulness
    type: percentage
    metric: UsefulnessAnalyzer.passed
    condition: "== True"
    min_percentage: 70.0
    severity: medium
    title: "Usefulness pass rate"
```

## Output Files

After running analysis, find results in your `output_path`:

| File | Description |
|------|-------------|
| `analysis.parquet` | Full results DataFrame |
| `analysis.csv` | CSV format (if requested) |
| `test_results.json` | Test pass/fail details |
| `summary.json` | Overview statistics |

## Dataset Formats

The analyzer supports multiple formats:

1. **Local JSONL files** - Set `dataset_path`
2. **HuggingFace datasets** - Set `dataset_name`

### Recommended Public Datasets

| Dataset | Format | Turns | Notes |
|---------|--------|-------|-------|
| `HuggingFaceH4/ultrachat_200k` | messages | Multi-turn (4-14) | Recommended for testing |
| `argilla/databricks-dolly-15k-curated-en` | instruction/response | Single-turn | Simpler format |
| `OpenAssistant/oasst2` | messages | Multi-turn | Human feedback data |

### Expected Conversation Format (Oumi native)

```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm doing well!"}
  ]
}
```

### Also Supported

- **Alpaca format**: `instruction`, `input`, `output` fields
- **Prompt/response pairs**: `prompt`/`response`, `question`/`answer`
- **Dolly format**: `original-instruction`, `original-response`

## Discover Available Metrics

```python
from oumi.analyze import list_metrics, generate_tests

# Show all analyzers and their metrics
list_metrics()

# Show metrics for specific analyzer
list_metrics("LengthAnalyzer")

# Generate example test config
yaml_config = generate_tests("LengthAnalyzer")
print(yaml_config)
```

## Troubleshooting

### "Unknown analyzer" error
Make sure you're using valid analyzer IDs. Run `--list-metrics` to see available analyzers.

### LLM analyzer not working
1. Check your API key is set: `echo $OPENAI_API_KEY`
2. Verify the model name is correct (e.g., `gpt-4o-mini`)
3. Start with `sample_count: 1` to test

### Token counting returns 0
Install tiktoken: `pip install tiktoken`
And specify the encoding: `tiktoken_encoding: cl100k_base`

### Import errors
Ensure you installed with analyze extras: `pip install -e ".[analyze]"`

## Architecture Overview

```
oumi.analyze/
├── base.py          # Base analyzer classes (Message/Conversation/Dataset/Preference)
├── pipeline.py      # AnalysisPipeline orchestration
├── config.py        # TypedAnalyzeConfig
├── cli.py           # CLI utilities
├── analyzers/
│   ├── length.py    # LengthAnalyzer
│   └── llm_analyzer.py  # LLM-based analyzers
├── results/
│   ├── length.py    # LengthMetrics model
│   └── llm_judgment.py  # LLMJudgmentMetrics model
└── testing/
    ├── engine.py    # TestEngine
    └── results.py   # Test result models
```

## Full Example Workflow

```bash
# 1. Install dependencies
pip install -e ".[analyze]"

# 2. Set API key (for LLM analyzers)
export OPENAI_API_KEY=sk-...

# 3. Run analysis
oumi analyze --config configs/examples/analyze/typed_llm_analyzer_example.yaml --typed

# 4. View results
cat analysis_output/llm_analyzer_demo/summary.json
```

```python
# Or via Python
from oumi.analyze import run_from_config_file, print_summary

results = run_from_config_file(
    'configs/examples/analyze/typed_llm_analyzer_example.yaml'
)
print_summary(results)

# Access DataFrame for further analysis
df = results['dataframe']
print(df.columns.tolist())
print(df.head())
```
