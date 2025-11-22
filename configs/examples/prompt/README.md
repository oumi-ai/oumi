# Prompt Optimization Examples

This directory contains example configuration files for the `oumi prompt` command, which uses state-of-the-art algorithms to optimize prompts, few-shot examples, and generation hyperparameters.

## Quick Start

First, install the prompt optimization dependencies:

```bash
pip install 'oumi[prompt-optimization]'
```

**For OpenAI models (GPT-4o-mini, GPT-4, etc.)**, tiktoken is now included and will be used for accurate token counting.

Then run optimization with one of the example configs:

```bash
oumi prompt -c configs/examples/prompt/mipro_basic.yaml
```

## Example Configurations

### 1. `mipro_basic.yaml`
Basic configuration using **MIPROv2** (Multi-prompt Instruction PRoposal Optimizer).

- **Best for**: Datasets with 300+ examples
- **Optimizes**: Prompt instructions and few-shot demonstrations
- **Algorithm**: MIPROv2 with Bayesian optimization
- **Speed**: Moderate (50 trials)

### 2. `gepa_basic.yaml`
Configuration using **GEPA** (Genetic-Pareto) optimizer.

- **Best for**: Complex tasks requiring reflective optimization
- **Optimizes**: Prompt instructions and demonstrations
- **Algorithm**: Reflective prompt evolution with Pareto selection
- **Speed**: Fast (30 trials, uses fewer rollouts than RL methods)

### 3. `bootstrap_basic.yaml`
Configuration using **BootstrapFewShot** optimizer.

- **Best for**: Small datasets (~10-50 examples)
- **Optimizes**: Few-shot example selection
- **Algorithm**: Bootstrap sampling with validation
- **Speed**: Very fast (20 trials)

### 4. `hyperparameter_optimization.yaml`
Advanced configuration with hyperparameter tuning.

- **Optimizes**: Prompts, demos, AND generation hyperparameters
- **Includes**: Temperature, top_p, max_tokens optimization
- **Best for**: Production use cases requiring cost/performance optimization

### 5. `mipro_gpt4o_mini.yaml`, `bootstrap_gpt4o_mini.yaml`, `gepa_gpt4o_mini.yaml`
Configurations using **OpenAI GPT-4o-mini** (cloud API).

- **Model**: OpenAI's latest small model - fast, cost-effective, intelligent
- **Requirements**: Set `OPENAI_API_KEY` environment variable
- **Cost**: ~$1-5 USD for typical optimization (50 trials, 100 examples)
- **Best for**: Production prompt optimization with high-quality results
- **Features**: Adaptive concurrency control, automatic rate limiting, retry logic

## Dataset Format

Your training and validation datasets should be in JSONL format with `input` and `output` fields:

```jsonl
{"input": "What is the capital of France?", "output": "Paris"}
{"input": "What is 2 + 2?", "output": "4"}
```

Alternatively, you can use Oumi Conversation objects.

## Supported Optimizers

- **mipro**: MIPROv2 - Data-aware instruction generation with Bayesian optimization
- **gepa**: GEPA - Reflective prompt evolution (outperforms RL methods)
- **bootstrap**: BootstrapFewShot - Simple few-shot example selection
- **evolutionary**: Evolutionary/genetic algorithms (custom implementation)

## Supported Metrics

- **accuracy**: Exact match accuracy (classification tasks)
- **f1**: Token-level F1 score
- **bleu**: BLEU score (requires `sacrebleu`)
- **rouge**: ROUGE-L score (requires `rouge-score`)
- **embedding_similarity**: Cosine similarity between embeddings (requires `sentence-transformers`)
- **bertscore**: BERTScore for semantic similarity (requires `bert-score`)
- **llm_judge**: Use an LLM to judge quality (requires `openai`, can be expensive)
- **custom**: Use your own metric function

## Command Line Overrides

You can override any config parameter via CLI:

```bash
oumi prompt -c configs/examples/prompt/mipro_basic.yaml \
  --optimization.num_trials=100 \
  --optimization.optimizer=gepa \
  --metric=f1
```

## Output

The optimization results are saved to the `output_dir` specified in your config:

- `optimized_prompt.txt` - The optimized instruction
- `optimized_demos.jsonl` - Selected few-shot examples
- `optimized_hyperparameters.json` - Optimized generation settings
- `optimization_results.json` - Full optimization history and metadata

## New Features

### Cost Estimation & Warnings ⚠️
The system now automatically estimates costs before running optimization and warns you if costs exceed thresholds:

```
COST ESTIMATE
================================================================================
Model: gpt-3.5-turbo
Estimated Cost: $15.50 USD
⚠️  HIGH COST WARNING: Consider reducing trials...
```

Costs are estimated for popular models (GPT-4, Claude, etc.). Local models show $0 cost.

### Checkpointing & Resume 💾
Long-running optimizations now save checkpoints automatically:

- Checkpoints saved every 5 minutes (configurable via `optimization.checkpoint_interval`)
- Automatically resumes from checkpoint if interrupted
- Disable with `optimization.enable_checkpointing: false`

If optimization is interrupted, simply rerun the same command - it will resume from the last checkpoint.

### Progress Tracking 📊
Real-time progress updates during DSPy optimization:

- Shows inference call counts during optimization
- Estimates remaining time
- Verbose mode (`optimization.verbose: true`) shows detailed progress

### Semantic Similarity Metrics 🎯
New metrics for better evaluation of generation tasks:

- `embedding_similarity`: Uses sentence transformers for semantic similarity
- `bertscore`: BERTScore for more nuanced evaluation
- `llm_judge`: Use GPT/Claude to judge response quality (experimental, expensive)

### Fast Mode (Skip Final Eval) ⚡
Skip redundant final evaluation to save time and cost:

```yaml
optimization:
  skip_final_eval: true  # Skip final validation re-evaluation
```

DSPy already evaluates during optimization, so this final evaluation provides verification but doubles evaluation costs. Enable this to save ~50% of evaluation inference calls.

## Performance Tips

1. **Start small**: Use `max_training_samples` and `max_validation_samples` to limit dataset size during initial experiments
2. **Choose the right optimizer**:
   - Small datasets (20-50 examples): Use `bootstrap`
   - Medium datasets (50-300 examples): Use `gepa`
   - Large datasets (300+ examples): Use `mipro`
3. **Use faster models**: Start with small models like SmolLM2-135M for quick iteration
4. **Enable verbose mode**: Set `optimization.verbose: true` to see progress
5. **Set a seed**: Use `optimization.seed` for reproducible results
6. **Watch your costs**: Check the cost estimate before starting expensive optimization runs

## References

- [DSPy Documentation](https://dspy.ai/)
- [MIPROv2 Paper](https://dspy.ai/api/optimizers/MIPROv2/)
- [GEPA Paper](https://arxiv.org/abs/2507.19457)
