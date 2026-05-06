# MCA (mcore_adapter) — Learnings & Quirks

## Core Architecture Difference: Label Alignment

**The most dangerous silent bug.** HuggingFace CausalLM shifts labels internally in `forward()`:
```python
loss = CE(logits[:, :-1, :], labels[:, 1:])
```
Megatron/MCA does **not** shift. It computes `CE(logits[:, i], labels[:, i])` directly. This means if you pass labels unmodified, every position is computing loss on the *wrong* target — the model learns "predict the token you already see" rather than "predict the next token."

**Symptom**: the model generates infinite `<think><think><think>...` loops. After the `<think>` token appears in input at position N, the unshifted label at position N is also `<think>`, so the model learns to always repeat it.

**Fix**: pre-shift at collator time by trimming `input_ids[:, :-1]` and `labels[:, 1:]` (and `attention_mask[:, :-1]`). This reduces sequence length by 1 and aligns `labels[i] = next token after input_ids[i]`. Both LlamaFactory and ROLL do this explicitly. We implemented this in `_collator_with_labels` in `McaSftTrainer`.

---

## Data Shuffling: SequentialSampler Is Hardcoded

MCA's training DataLoader **always** uses `SequentialSampler` (hardcoded at `mcore_adapter/trainer/trainer.py:189`). It never shuffles, regardless of any config setting. The warning `"Currently, train dataloader drop_last must be set to True!"` is the only hint that something unusual is happening.

**Symptom**: if your dataset is sorted by class (which is common — e.g., banking77 has all class-0 examples first, then all class-1, etc.), the model sees batches of identical labels for many consecutive steps. It quickly overfits to the first class, then catastrophically forgets it as later classes appear. Result: mode collapse to 1–3 predicted classes, ~1% accuracy.

**How it manifests in training metrics**:
- Without shuffle: initial loss is artificially low (~0.59 instead of ~2.6) because the first batch is a trivially easy single-class problem. Grad norms spike to 100–600+ as the model thrashes between class clusters.
- With shuffle: initial loss is ~2.6 (correct for 77-class classification on a random model), grad norms stabilize to 9–15 and decay normally.

**Fix**: call `train_dataset.shuffle(seed=seed)` before passing to `McaTrainer`. We added this in `McaSftTrainer.__init__`. If the dataset doesn't have a `.shuffle()` method, you need to pre-shuffle the data file.

---

## Tied Embeddings: `lm_head.weight` Missing from Checkpoint

mcore_adapter's `convert_checkpoint_to_hf` converter **omits** `lm_head.weight` when `config.json` has `tie_word_embeddings=True` (since the weight is mathematically identical to `embed_tokens.weight`). HuggingFace handles this transparently at load time. **vLLM does not** — it requires the key to be physically present in the safetensors file.

**Symptom**: vLLM loads the model without error but produces garbage output or crashes on inference.

**Affected models**: Qwen3, Qwen2.5, and any other model with `tie_word_embeddings: true` in config.json.

**Fix**: after conversion, load the safetensors file, check if `lm_head.weight` is missing but `model.embed_tokens.weight` is present, then clone it: `tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()`. The `.clone()` is **required** — safetensors cannot serialize tensors that share underlying storage.

We implemented `_fix_tied_embeddings()` in `McaSftTrainer.save_model()`.

---

## `calculate_per_token_loss` Flag Is Confusing

The `McaTrainingArguments.calculate_per_token_loss` setting (default `False`) affects **only metrics logging**, not the gradient. The actual gradient normalization is controlled by `TransformerConfig.calculate_per_token_loss` (a separate Megatron-Core model config field), which MCA never sets.

Since `model.config.calculate_per_token_loss` is always `False`, Megatron's `forward_step` **always** divides the loss by `num_active_tokens × num_microbatches` before backprop. In other words, the default MCA gradient **is** per-token normalized — you do not need to set this flag for correct gradient scale.

Setting `calculate_per_token_loss: true` in `mca_config_overrides` only changes whether the logged `loss` metric is computed as a weighted average (sum/count) across microbatches vs a simple mean of pre-normalized microbatch averages. The effective logged value is nearly identical either way.

**Don't add this to configs thinking it fixes gradient instability** — it doesn't.

---

## `warmup_ratio` Is Deprecated in MCA's Underlying HF Version

MCA's trainer is built on a version of `transformers` that has deprecated `warmup_ratio` in favor of `warmup_steps`. You'll see this warning:
```
warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
```
This is harmless — the value still works — but worth being aware of. The `McaSftTrainer._build_mca_args()` passes `warmup_ratio` if set, and the warning is from MCA's internals.

---

## Model Construction: MCA Requires a Local Path

`McaAutoModel.from_pretrained()` internally calls `os.path.isfile(config.json)` — it does not accept HuggingFace Hub IDs directly. If you pass a Hub ID like `"Qwen/Qwen3-0.6B"`, MCA will fail with a path error.

**Fix**: always call `snapshot_download()` first to get a local path. We do this in `_resolve_model_path()` in `McaSftTrainer`.

---

## Parallelism Validation: WORLD_SIZE Must Be Divisible by TP × PP × EP

MCA will initialize silently and then hang or crash if `WORLD_SIZE % (TP × PP × EP) != 0`. The remaining GPUs are used for data parallelism. We validate this upfront in `_validate_world_size()` to give a clear error.

---

## Checkpoint Format: FSDP/DeepSpeed Checkpoints Are Incompatible

MCA's Megatron-style checkpoints (`model_optim_rng.pt`, `optimizer.pt` inside `iter_XXXXXXX/mp_rank_00/`) are **not compatible** with HF checkpoints from FSDP or DeepSpeed trainers. Attempting to resume from one will fail silently or produce corrupt weights.

We validate this in `_validate_checkpoint_format()` by checking for FSDP (`model_world_size_*_rank_*.pt`) and DeepSpeed (`*/mp_rank_*/model_states.pt`) files.

---

## `transformer_impl="local"` vs `"transformer_engine"` Quality Warning

MCA supports two transformer backends:
- `"local"`: Megatron-Core's own reimplementation of the transformer. **May produce different fine-tuning outcomes** from HuggingFace's native implementation, especially for edge cases in attention and normalization.
- `"transformer_engine"`: NVIDIA Transformer Engine. Produces results much closer to HuggingFace. Requires the `transformer_engine` package.

For production use where quality parity with TRL matters, prefer `transformer_engine`. The `"local"` backend is faster to set up but may diverge.

---

## Distributed Init: Environment Variables Must Be Set Manually

MCA manages its own distributed initialization using `torch.distributed`. You must set `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` before invoking the trainer. These aren't set automatically by `oumi train`. For single-GPU:
```bash
MASTER_ADDR=localhost MASTER_PORT=29500 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  oumi train -c ...
```
For multi-GPU, `oumi distributed torchrun` handles this.

---

## `gradient_checkpointing` Is Not Supported

MCA's `VirtualModels` wrapper doesn't support HuggingFace's `gradient_checkpointing_enable()`. Setting `gradient_checkpointing: true` in the training config will be silently ignored or crash. For activation recomputation, use Megatron's native mechanism instead:
```yaml
megatron:
  recompute_granularity: "full"  # or "selective"
```

---

## MCA Logs Metrics Differently

MCA doesn't write `trainer_state.json` in the standard HF format until after training completes. During training, metrics are printed to stdout as Python dict strings embedded in the tqdm progress bar:
```
{'loss': '0.4194', 'grad_norm': '14.98', 'learning_rate': '1.129e-05', ...}
```
These are not JSON — they're Python repr. The `trainer_state.json` file only appears after the full run finishes. Plan accordingly if you're monitoring live metrics.

---

## Summary Table

| Issue | Symptom | Fix |
|-------|---------|-----|
| No label shift | `<think>` infinite loop; degenerate output | Pre-shift labels in collator: `input_ids[:, :-1]`, `labels[:, 1:]` |
| No data shuffle (SequentialSampler) | Mode collapse, 1% accuracy on sorted datasets | `train_dataset.shuffle(seed)` before passing to McaTrainer |
| `lm_head.weight` missing | vLLM garbage output on tied-embedding models | Clone `embed_tokens.weight` → `lm_head.weight` in safetensors after conversion |
| Hub ID not accepted | MCA crash at model load | `snapshot_download()` to get local path first |
| `gradient_checkpointing` silently broken | No memory savings | Use `recompute_granularity` instead |
| FSDP/DeepSpeed checkpoint incompatibility | Silent corruption on resume | Validate checkpoint format; convert to HF first |
| `transformer_impl="local"` quality divergence | Different results from TRL | Use `"transformer_engine"` for production quality parity |
