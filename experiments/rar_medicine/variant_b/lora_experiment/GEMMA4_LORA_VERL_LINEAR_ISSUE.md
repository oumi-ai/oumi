# Support Gemma 4 LoRA rollouts and weight synchronization in VERL GRPO

## Summary

Enable LoRA GRPO for Gemma 4 models using VERL with vLLM rollouts.

LoRA training itself works in Hugging Face Transformers/PEFT. The blocker is
the rollout side: vLLM can accept and register a Gemma 4 adapter while silently
zeroing the adapter buffers during activation. After the actor's first effective
optimizer update, the Hugging Face actor contains a nonzero LoRA update but
vLLM continues generating from the base model. The run continues without an
exception, but it is no longer standard on-policy GRPO.

The immediate target is `google/gemma-4-E2B-it` in
`experiments/rar_medicine/variant_b/train_verl.yaml`. The solution should be
applicable to other text-only Gemma 4 training and inference paths where
possible.

## Why Gemma 4 plus vLLM LoRA is difficult

There are several independent compatibility layers:

1. **Model-class support.** vLLM 0.19.x has a native Gemma 4 implementation,
   but `Gemma4ForConditionalGeneration` was not wired into vLLM's LoRA
   interface. That integration was added upstream in
   [vLLM PR #39291](https://github.com/vllm-project/vllm/pull/39291) and shipped
   after 0.19.x.
2. **Multimodal versus text-only architectures.** Transformers trains the
   conditional-generation wrapper, which includes language, vision, and audio
   components. A text-only vLLM rollout may use `Gemma4ForCausalLM`. Adapter
   keys and supported target modules must be mapped between those structures;
   unused vision/audio keys must either be supported or removed explicitly.
3. **Packed projections.** PEFT saves separate `q_proj`, `k_proj`, `v_proj`,
   `gate_proj`, and `up_proj` tensors. vLLM may execute packed `qkv_proj` and
   `gate_up_proj` modules. The adapter loader must pack, shard, and scale these
   tensors correctly.
4. **YOCO module aliasing.** Gemma 4 constructs its decoder layers once and
   then exposes the same physical layer objects under `layers` and under
   `self_decoder.decoder_layers` or `cross_decoder.decoder_layers`. vLLM
   discovers both names with `named_modules(remove_duplicate=False)`.
5. **Dynamic RL synchronization.** Static serving only needs to load one
   adapter once. GRPO updates the adapter continually and requires the rollout
   engine to use the new values before every subsequent generation batch.
6. **The failure is silent.** Adapter parsing, registration, weight-transfer
   timing, and generation can all succeed while the effective adapter remains
   zero. A successful `LoRARequest` is therefore not an end-to-end correctness
   check.

## Root cause: alias activation erases the adapter

The same physical projection can be reachable through two names:

```text
language_model.model.layers.0.self_attn.qkv_proj
language_model.model.self_decoder.decoder_layers.0.self_attn.qkv_proj
```

Conceptually:

```python
Q = one_physical_qkv_projection

modules = {
    "model.layers.0.self_attn.qkv_proj": Q,
    "model.self_decoder.decoder_layers.0.self_attn.qkv_proj": Q,
}
```

The PEFT adapter contains a weight under the canonical `layers.0` path. It does
not contain a duplicate weight under the YOCO alias because both names refer to
one physical projection.

Unpatched vLLM activates an adapter by making a decision for every name:

```python
for module_name, module in self.modules.items():
    module_lora = get_lora_weights(adapter, module_name)
    if module_lora is None:
        module.reset_lora(slot)
    else:
        module.set_lora(slot, module_lora.A, module_lora.B)
```

For the example above, activation becomes:

```text
canonical name has weights  -> Q.set_lora(slot, A, B)
alias name has no weights   -> Q.reset_lora(slot)
```

Both calls mutate the same `Q`. The second call zeros `lora_a_stacked` and
`lora_b_stacked`, so generation behaves like the base model even though the
adapter is registered. This behavior is tracked in
[vLLM issue #41754](https://github.com/vllm-project/vllm/issues/41754) and the
more direct aliasing fix is proposed in
[vLLM PR #39816](https://github.com/vllm-project/vllm/pull/39816).

Local probing on Gemma 4 E2B found 422 LoRA-capable names for approximately 212
physical linear modules; 210 physical modules were registered under at least
two names. A vLLM+LoRA generation was identical to the vLLM base generation,
while Transformers+PEFT using the same adapter differed from its base model.

This is not specific to VERL. It affects ordinary vLLM runtime adapter serving
as well. VERL makes it a training-correctness problem because it relies on the
same activation mechanism after synchronizing each updated adapter.

## Background: how LoRA weight synchronization works in GRPO

There are four relevant policies:

| Policy | Purpose |
|---|---|
| Actor/current policy | Hugging Face/PEFT model being optimized |
| Rollout policy | vLLM model that generates responses |
| Old policy | Actor snapshot used in the PPO/GRPO probability ratio |
| Reference policy | Frozen base policy used for the KL penalty |

For LoRA, the effective actor weight is:

```text
W_effective = W_base + (alpha / rank) * B @ A
```

Only `A` and `B` are trainable. The intended loop is:

```text
1. vLLM generates responses with the current LoRA.
2. VERL computes rewards and group-relative advantages.
3. The actor recomputes old_log_probs for those response tokens.
4. GRPO updates the actor's A/B tensors.
5. VERL extracts the new adapter tensors.
6. TensorLoRARequest transfers them to vLLM.
7. vLLM packs/shards the tensors and activates them in a GPU LoRA slot.
8. The next rollout uses W_base + the updated LoRA.
```

A new conventional LoRA starts with a zero effective update, commonly because
`A` is random and `B` is zero. The first rollout can therefore be valid even if
vLLM ignores the adapter:

```text
initial actor   = W_base + 0
initial rollout = W_base
```

After the first optimizer step that changes the adapter:

```text
actor           = W_base + delta_W_1
expected vLLM   = W_base + delta_W_1
broken vLLM     = W_base
```

The next responses are sampled from the base model, but VERL recomputes
`old_log_probs` using the updated actor. The normal PPO ratio compares the new
actor with that old actor; it does not automatically correct for a third,
stale behavior policy that actually generated the samples.

`rollout.calculate_log_probs: true` helps diagnose this. It makes vLLM return
the probabilities it assigned to the sampled tokens, and VERL compares them
with the actor's recomputed probabilities. By default, these rollout
probabilities are diagnostic and are not substituted for actor
`old_log_probs` in the loss.

The useful W&B guards are:

- `training/rollout_probs_diff_mean`: should remain near its calibrated BF16
  baseline.
- `training/rollout_actor_probs_pearson_corr`: should remain near 1.
- `rollout_corr/training_log_ppl` and `rollout_corr/rollout_log_ppl`: should
  remain overlaid.
- `rollout_corr/log_ppl_abs_diff` and `rollout_corr/k3_kl`: should remain near
  zero.
- `rollout_corr/ppl_ratio`: should remain near 1.

On the repaired full-weight Gemma 4 path, the observed healthy baseline was
approximately `rollout_probs_diff_mean = 0.007`, Pearson correlation
`0.998-0.999`, and actor/vLLM log-PPL values within approximately
`0.002-0.003`. Exact thresholds should be calibrated for the final vLLM and
kernel versions.

## How `agent-finetune` avoids the problem

The [`oumi-ai/agent-finetune`](https://github.com/oumi-ai/agent-finetune/tree/main)
repository demonstrates successful Gemma 4 E2B LoRA SFT, but it does not use
vLLM to execute that adapter.

Its flow is:

```text
TRL SFT + Transformers/PEFT
            |
            v
PEFT adapter_model.safetensors
            |
            v
llama.cpp convert_lora_to_gguf.py
            |
            v
GGUF adapter served by llama-server --lora-scaled
            |
            v
evaluation using GEMMA_RAW and POST /lora-adapters
```

Relevant references:

- The training recipe uses `trainer_type: TRL_SFT` and `use_peft: true`:
  [`gemma4_e2b_lora_xs.yaml`](https://github.com/oumi-ai/agent-finetune/blob/3e81f5b52623ed2244296b3cc1de0cf69d01426f/training/gemma4_e2b_lora_xs.yaml#L50-L121).
- The adapter is converted to GGUF and evaluated through llama.cpp:
  [`VERIFY_LORA.md`](https://github.com/oumi-ai/agent-finetune/blob/3e81f5b52623ed2244296b3cc1de0cf69d01426f/experiments/2026-08-04-distill-main/VERIFY_LORA.md#L76-L135).
- The server is `llama-server` and uses `--lora-scaled`:
  [`serve_local_mac.sh`](https://github.com/oumi-ai/agent-finetune/blob/3e81f5b52623ed2244296b3cc1de0cf69d01426f/experiments/2026-08-04-distill-main/scripts/serve_local_mac.sh#L66-L92).
- Its custom evaluation client switches adapters through llama.cpp's
  `/lora-adapters` endpoint:
  [`harness/gemma_raw.py`](https://github.com/oumi-ai/agent-finetune/blob/3e81f5b52623ed2244296b3cc1de0cf69d01426f/harness/gemma_raw.py#L210-L217).

The repository has YAML blocks named `inference_engine`, but the fine-tuned
Gemma recipe uses `type: GEMMA_RAW`, not Oumi's `engine: VLLM`:
[`vyra_gemma_ft.yaml`](https://github.com/oumi-ai/agent-finetune/blob/3e81f5b52623ed2244296b3cc1de0cf69d01426f/harness/recipes/vyra_gemma_ft.yaml#L38-L54).

Its one concrete `vllm serve` example loads a fully materialized BF16 model and
does not pass an adapter. Therefore, `agent-finetune` validates Transformers
LoRA training and llama.cpp LoRA inference, but it never exercises vLLM's
Gemma 4 LoRA manager or per-step VERL synchronization.

## How Gemma 4 31B onboarding handled the problem

Gemma 4 31B onboarding in `oumi-ai/api` implemented a downstream, text-only
workaround at pinned commit
[`ddd25df`](https://github.com/oumi-ai/api/blob/ddd25dfd5a9df719ff8fb98b54166754005946ce/shared/src/shared/model_setup_overrides.py#L109-L175).

The setup has four relevant parts:

1. **Pin a validated dependency set and model revision.** Training uses
   Transformers 5.8.0; inference pins vLLM 0.19.1 and Transformers 5.8.0. The
   Hugging Face revision is pinned to a pre-schema-change commit so training
   and serving resolve the same model structure and chat template.
2. **Redirect the vLLM model registry.** Because vLLM 0.19.1 does not expose
   LoRA through `Gemma4ForConditionalGeneration`, the setup rewrites its
   registry entry to instantiate the text-only `Gemma4ForCausalLM`. This is the
   compatibility gap later addressed upstream by vLLM PR #39291.
3. **Remove adapter tensors for modules absent from the serving model.** The
   training preset uses `all-linear` against the multimodal conditional model,
   so the generated adapter includes vision keys. The pre-inference script
   `fix_gemma4_lora_adapter.py` removes `vision_tower` and `embed_vision` keys.
   Text keys remain unchanged because vLLM's HF-to-vLLM mapper handles their
   language-model prefix.
4. **Patch the YOCO alias bug.** The setup modifies vLLM's adapter activation
   loop to keep a set of `id(module)` values and skip later names that point to
   an already-visited physical module:

```python
seen_modules = set()
for module_name, module in self.modules.items():
    if id(module) in seen_modules:
        continue
    seen_modules.add(id(module))
    # existing set_lora/reset_lora logic
```

This allowed the canonical module visit to install LoRA weights without a later
YOCO alias resetting the same buffer. It is a strong precedent because it
solved actual Gemma 4 31B runtime LoRA serving on the platform.

However, the exact downstream patch assumes the name containing adapter weights
is encountered before its aliases. That is currently true for the validated
Gemma 4 layout because `layers` is registered before the decoder wrappers, but
it is an ordering dependency. A general implementation should inspect every
alias first and prefer a matching weight rather than blindly retaining the
first name.

The 31B solution also targets text-only serving. It does not by itself validate
VERL's in-memory `TensorLoRARequest` path, repeated adapter replacement, FSDP
extraction, or actor/rollout probability agreement during GRPO.

## Proposed approaches

### Approach 1: use native conditional-model LoRA plus the robust upstream alias fix

**Recommended long-term approach.** Use a vLLM version containing the
`Gemma4ForConditionalGeneration` LoRA integration from PR #39291, and apply or
backport the robust form of PR #39816 until it is available in a compatible
release.

Activation should be two-pass and keyed by physical module identity:

```python
chosen = {}
for module_name, module in self.modules.items():
    weights = get_lora_weights(adapter, module_name)
    module_id = id(module)
    if module_id not in chosen or (
        weights is not None and chosen[module_id].weights is None
    ):
        chosen[module_id] = Choice(module, weights)

for module, weights in chosen.values():
    if weights is None:
        module.reset_lora(slot)
    else:
        module.set_lora(slot, weights.A, weights.B)
```

This retains all names for adapter matching while applying `set_lora` or
`reset_lora` once per physical module, independently of traversal order.

Advantages:

- Fixes the underlying vLLM bug rather than relying on artifact rewriting.
- Works for serving and VERL synchronization.
- Preserves valid canonical and prefixed adapter naming conventions.
- Can be covered by a small model-manager regression test with reversed alias
  order.

Risks/work:

- Requires a compatible vLLM/Transformers/Torch version set for Oumi and VERL.
- Packed projection, tensor-parallel, expert/MoE, and multimodal target mappings
  still require end-to-end validation.
- The open upstream patch may need to be carried temporarily.

### Approach 2: extend the proven Gemma 4 31B workaround to E2B GRPO

Use the existing platform workaround as the shortest path for the current
text-only experiment:

1. Pin the validated vLLM 0.19.1 stack.
2. Redirect `Gemma4ForConditionalGeneration` to `Gemma4ForCausalLM`.
3. Remove unsupported vision/audio adapter keys when necessary.
4. Apply the `id(module)` activation dedupe patch.
5. Validate VERL's repeated tensor synchronization path.

Advantages:

- Reuses a platform path already proven for Gemma 4 31B serving.
- Avoids a broader dependency upgrade.
- Small and reversible downstream patch.

Risks/work:

- Text-only workaround rather than full conditional-model support.
- The current first-seen dedupe is ordering-dependent; use the robust two-pass
  variant for new work.
- E2B has a different scale/MoE structure, so success on 31B is evidence but
  not sufficient validation.
- The GRPO tensor request may use different names from a disk-loaded PEFT
  adapter.

### Approach 3: carry a narrowly scoped Oumi/VERL runtime patch

Patch `LoRAModelManager.activate_adapter` during the job setup or through a
maintained compatibility module, pin the exact vLLM commit/version, and add an
E2E test to Oumi.

A concrete local spike using the robust two-pass form, together with an
isolated LoRA VERL config, is under
`experiments/rar_medicine/variant_b/lora_experiment/`.

This can unblock the experiment before the upstream fix is released, while
avoiding changes to Gemma 4's model graph. The patch should use two-pass
weight-preferred deduplication, fail loudly if no requested adapter tensors are
installed, and report counts of matched names, unique physical modules, and
nonzero activated buffers.

The main cost is maintenance: site-package/runtime patches are sensitive to
vLLM refactors and must be guarded by source/version checks.

### Approach 4: avoid runtime LoRA in the rollout engine

Available fallbacks are:

- Continue with full fine-tuning and full-weight synchronization, which is the
  current working GRPO path.
- Merge the adapter into the base model for post-training evaluation and serve
  the resulting full checkpoint.
- Evaluate an alternative rollout backend that supports Gemma 4 and live LoRA
  replacement, then add the corresponding VERL synchronization integration.

Merging is useful for static inference but is not an attractive per-step GRPO
solution. Re-merging and transferring the full model after every optimizer step
eliminates most of LoRA's memory and synchronization advantages. The
`agent-finetune` llama.cpp path is similarly appropriate for evaluation, but it
does not currently provide a VERL actor-to-rollout tensor-sync implementation.

### Approach 5: duplicate or rewrite adapter keys for every alias

It may be possible to manufacture matching weights for every canonical and
YOCO alias name, preventing the alias visit from calling `reset_lora`.

This is not recommended as the primary solution:

- It encodes vLLM's internal module traversal into adapter artifacts.
- It can duplicate large amounts of adapter metadata/tensors.
- It is fragile across model and vLLM versions.
- It does not generalize as cleanly as applying one activation decision per
  physical module.

## Recommended implementation plan

1. Start with Approach 1 if the vLLM/Torch upgrade fits Oumi's dependency
   constraints; otherwise use Approach 2 with the robust two-pass dedupe.
2. Add a minimal offline activation test before starting GRPO:
   - load Gemma 4 base in vLLM;
   - install a deliberately nonzero PEFT-shaped adapter;
   - verify targeted vLLM LoRA buffers remain nonzero after activation;
   - verify vLLM+LoRA logits/output differ from vLLM base;
   - verify vLLM+LoRA follows Transformers+PEFT.
3. Exercise the exact VERL `TensorLoRARequest` path twice with different adapter
   tensors to prove that replacement works, not just first-time loading.
4. Run a short GRPO smoke test from a zero-initialized adapter. Confirm the
   actor changes, synchronize, and inspect the following rollout step.
5. Keep rollout log-probability diagnostics enabled and enforce a calibrated
   actor/vLLM agreement guard.
6. Run a resume test from a nonzero adapter; this removes the initial
   zero-adapter grace period.
7. Document the supported model class, target modules, package pins, and any
   deliberate text-only limitations.

## Acceptance criteria

- [ ] Gemma 4 E2B LoRA GRPO starts with the documented package versions and no
      unsupported-model error.
- [ ] The solution supports the standard decoder targets: `q_proj`, `k_proj`,
      `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`, including
      vLLM's packed projection mapping.
- [ ] Adapter activation makes one final `set_lora`/`reset_lora` decision per
      physical module, independent of alias traversal order.
- [ ] A nonzero adapter produces a measurable vLLM base-versus-LoRA logits or
      output difference and agrees with Transformers+PEFT on the same inputs.
- [ ] Two consecutive VERL adapter synchronizations install the second set of
      tensors; the rollout engine does not retain the first adapter or revert to
      the base model.
- [ ] After the first effective optimizer update, the next rollout reflects the
      updated actor.
- [ ] Actor/vLLM W&B metrics remain near the calibrated healthy baseline for a
      multi-step smoke run; they do not trend toward the stale-base signature.
- [ ] Resume from a nonzero LoRA checkpoint is covered.
- [ ] Missing, ignored, reset, or all-zero requested adapter tensors cause a
      clear failure or high-signal error rather than silent base-model output.
- [ ] The implementation includes a regression test for canonical and aliased
      names referring to the same module, including reversed name order.
- [ ] Any text-only restriction and vision/audio adapter handling are explicit
      in configuration and documentation.

## References

- Current experiment:
  `experiments/rar_medicine/variant_b/train_verl.yaml`
- Detailed GRPO and W&B explanation:
  `experiments/rar_medicine/variant_b/LORA_GRPO_SYNC_EXPLAINER.md`
- [vLLM Gemma 4 model implementation](https://github.com/vllm-project/vllm/pull/38826)
- [vLLM conditional-generation LoRA integration #39291](https://github.com/vllm-project/vllm/pull/39291)
- [vLLM Gemma 4 alias-reset fix #39816](https://github.com/vllm-project/vllm/pull/39816)
- [vLLM ignored Gemma 4 adapter issue #41754](https://github.com/vllm-project/vllm/issues/41754)
- [Gemma 4 31B platform workaround](https://github.com/oumi-ai/api/blob/ddd25dfd5a9df719ff8fb98b54166754005946ce/shared/src/shared/model_setup_overrides.py#L109-L175)
- [`agent-finetune` LoRA verification path](https://github.com/oumi-ai/agent-finetune/blob/3e81f5b52623ed2244296b3cc1de0cf69d01426f/experiments/2026-08-04-distill-main/VERIFY_LORA.md)
