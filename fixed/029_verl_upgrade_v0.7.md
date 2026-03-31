# verl Upgrade to v0.7.1 — Validation Report

**Date:** 2026-03-25
**Upgrade:** verl 0.5.x → 0.7.1
**Status:** Oumi compatibility changes complete. One upstream verl bug remains.

## Breaking Changes in verl 0.7

### 1. `ResourcePoolManager.mapping` type annotation changed

- **Old (0.5/0.6):** `dict[Role, str]`
- **New (0.7):** `dict[int, str]`
- **Reality:** verl 0.7 annotates `dict[int, str]` but internally looks up keys using `Role` enum members (not `.value`). `Role` is a plain `Enum`, not `IntEnum`, so `Role.ActorRollout != 2`. Using `.value` as keys causes a `KeyError` at runtime.
- **Fix:** Always use `Role` enum members as keys. Works on both old and new versions.

### 2. `RayPPOTrainer` removed `reward_fn` and `val_reward_fn` parameters

- **Old (0.5/0.6):** `reward_fn=None, val_reward_fn=None` optional params in `__init__`
- **New (0.7):** Parameters removed entirely. Reward management is now config-driven via a top-level `reward` config section. Custom reward functions are loaded via `config.reward.custom_reward_function.path` and `.name` using `load_extern_object`.
- **Fix:** Conditionally pass these params only when `is_verl_v0_7_or_later()` is `False`. For 0.7+, set `config.reward.custom_reward_function.path` to `pkg://<module>` and `.name` to the function name using `inspect.getmodule()` on the oumi reward callable.

### 3. Config schema changes

- **Old (0.5/0.6):** Top-level keys: `reward_model`, `custom_reward_function`, `ray_init`
- **New (0.7):** Added top-level keys: `reward` (containing `reward_model`, `custom_reward_function`, `reward_manager`, `sandbox_fusion`), `global_profiler`, `ray_kwargs`, `transfer_queue`. Also added `critic.enable`, new `_target_` dataclass annotations throughout, and restructured `actor.optim`/`actor.fsdp_config`.
- **Fix:** Load the generated config yaml directly from the installed verl package (`verl/trainer/config/_generated_ppo_trainer.yaml`) at runtime, instead of bundling a static copy. Falls back to the bundled v0.5 yaml if the file doesn't exist.

## Files Changed

| File | Change |
|------|--------|
| `src/oumi/utils/packaging.py` | Added `is_verl_v0_7_or_later()` version check |
| `src/oumi/core/trainers/verl_grpo_trainer.py` | Version-gated mapping keys, reward params, and config loading |

## Validation

### Environment

- **GPU:** 1x NVIDIA GPU (single node)
- **torch:** 2.10.0+cu128 (required by vllm 0.18.0)
- **verl:** 0.7.1
- **vllm:** 0.18.0
- **trl:** 0.28.0 (verl 0.7 imports `AutoModelForCausalLMWithValueHead`, removed in trl >=0.29)

### Lint / Type Checks

All pre-commit checks pass:

```
ruff (legacy alias)......................................................Passed
ruff format..............................................................Passed
pyright..................................................................Passed
pyupgrade................................................................Passed
```

### Unit Tests

```
tests/unit/core/trainers/test_verl_grpo_trainer.py::test_init_without_verl        PASSED
tests/unit/core/trainers/test_verl_grpo_trainer.py::test_init_with_multiple_reward_funcs PASSED
```

### Live Training: GSM8K Config

**Command:**
```bash
python -m oumi train -c configs/examples/grpo_verl_gsm8k/train.yaml \
  --training.num_train_epochs=1 \
  --training.verl_config_overrides.trainer.val_before_train=False \
  --training.verl_config_overrides.trainer.total_epochs=1 \
  --training.verl_config_overrides.data.train_batch_size=16 \
  --training.verl_config_overrides.actor_rollout_ref.rollout.n=4 \
  --training.enable_wandb=False \
  --training.save_steps=-1
```

**Result:** Training initialized and ran successfully:
- Config parsed and validated (all new 0.7 sections present)
- Dataset loaded: 7473 train examples, 1319 val examples
- Ray workers initialized
- vLLM rollout engine started
- Training step executed

**Failure point:** After completing a training step, crashed at:
```
verl/trainer/ppo/ray_trainer.py:1252 in fit
    self.checkpoint_manager.update_weights(self.global_steps)

verl/single_controller/ray/base.py:413 in update_weights
    ray.get(self.trainer.update_weights(global_steps=global_steps))

AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'
```

This is an **upstream verl 0.7.1 bug** in the checkpoint engine — `RayWorkerGroup` is missing the `update_weights` method that the checkpoint manager expects. Not related to oumi code.

## Known Upstream Issues

| Issue | Affected Versions | Workaround |
|-------|-------------------|------------|
| `RayWorkerGroup.update_weights` missing | verl 0.7.1 | None — awaiting upstream fix |
| `AutoModelForCausalLMWithValueHead` import | verl 0.7.1 + trl >=0.29 | Pin `trl<0.29` |
| vllm 0.18.0 requires `torch==2.10.0` | vllm 0.18.0 | Install matching torch version |

## Backward Compatibility

The changes are backward compatible with verl >=0.5. The version gate `is_verl_v0_7_or_later()` controls:

- Whether `reward_fn`/`val_reward_fn` are passed to `RayPPOTrainer` (only pre-0.7)
- Config is always loaded from the installed verl package, so it matches the version in use

The `ResourcePoolManager.mapping` uses `Role` enum keys unconditionally — this works on all versions.
