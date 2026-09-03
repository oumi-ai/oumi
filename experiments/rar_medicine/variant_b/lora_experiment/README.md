# Gemma 4 LoRA/GRPO alias-patch experiment

This directory is an isolated LoRA copy of the current Variant B VERL config.
It tests a downstream fix for vLLM's Gemma 4 adapter-activation bug without
changing `../train_verl.yaml`.

## Files

- `vllm_gemma4_lora_alias.patch` changes `LoRAModelManager.activate_adapter()`
  from a mutate-as-it-traverses loop to two passes grouped by `id(module)`.
  If any name for a physical module has adapter weights, that choice wins over
  aliases without weights. Each physical LoRA buffer is then set or reset once.
- `train_verl.yaml` preserves the current full-FT config's batching, memory
  settings, and Gemma 4 actor fixes, and reinstates LoRA r=16 / alpha=32,
  attention-and-MLP targets, the tower exclusion regex, LR 1e-5, and vLLM's
  four fused `lora_target_modules`.
- `run.sh` applies the source patch idempotently to vLLM in the active Python
  environment, then launches this config with the parent experiment's reward
  registration and external actor fixes.

## Run

Activate the same environment that supplies `oumi`, `verl`, and vLLM 0.19.1,
then run from the repository root:

```bash
bash experiments/rar_medicine/variant_b/lora_experiment/run.sh
```

This patch fixes only alias-safe adapter activation. A stock vLLM 0.19.1 still
needs the existing Gemma 4 `SupportsLoRA`/target-module backport described in
`../RUN_REPORT.md`; the current `oumi` environment already contains that work.
On a different environment, verify model-class LoRA support separately.

The launcher refuses to continue if the installed source matches neither the
unpatched nor patched hunk. `SKIP_VLLM_PATCH=1` is available for an environment
that already contains an equivalent upstream fix; do not use it merely to
bypass a failed source check.

To reverse this exact downstream patch manually:

```bash
VLLM_SITE="$(python -c 'from pathlib import Path; import vllm; print(Path(vllm.__file__).resolve().parent.parent)')"
patch --reverse -p1 -d "${VLLM_SITE}" < experiments/rar_medicine/variant_b/lora_experiment/vllm_gemma4_lora_alias.patch
```

## First validation signal

The config enables `rollout.calculate_log_probs`. After the first optimizer
update, inspect these WandB metrics together:

- `training/rollout_probs_diff_mean` should stay near its pre-update numeric
  noise floor rather than grow with adapter training.
- `training/rollout_probs_diff_max` should remain bounded rather than jumping
  after every sync.
- `training/rollout_actor_probs_pearson_corr` should remain close to 1.

These metrics are a sync diagnostic, not a proof by themselves. A short smoke
run should also confirm that adapter tensors are nonzero after the optimizer
step and that vLLM output changes for a fixed prompt when the adapter is active.
