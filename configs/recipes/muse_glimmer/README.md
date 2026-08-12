# Muse Glimmer

## Summary

Configs for Meta Superintelligence Lab's Muse Glimmer model family. See the
[model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) for more information.
Models in this family include:

- [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) (dense, ~29.6B, image + text, 131K context) — **LoRA config available**

Muse Glimmer requires `transformers >= 5.15.0`, which must be upgraded manually
(`uv pip install -U "transformers>=5.15"`) because oumi currently pins
`transformers<5.10`. The architecture is absent from transformers 5.14.1 and earlier.

Full fine-tuning is not provided: at ~29.6B parameters the model does not fit FFT at
8K context on a single 8xH100 node.

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) if you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.
4. (Optional) If you wish to do deeper experimentation, follow our [instructions](https://oumi.ai/docs/en/latest/development/dev_setup.html) to clone the Oumi repository locally.
   - Make sure to delete the `oumi://` prefix when running Oumi commands, to disable fetching the latest configs from GitHub!

## Example Commands

### LoRA Training

LoRA is scoped to the language-model layers only. The ViT-G/14 perception encoder
reuses the text model's projection names (`q_proj`, `k_proj`, `v_proj`), so the recipe
targets the plain projection names and sets `lora_exclude_modules` to
`[".*vision_tower.*", ".*vision_adapter.*", ".*vision_projection.*"]` to keep LoRA off
the encoder and the modules that bridge it into the text hidden space. Oumi passes this
list to PEFT's `exclude_modules`.

To launch Muse Glimmer 30B LoRA training locally with FSDP (needs a multi-GPU node):

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/muse_glimmer/sft/30b_lora/train.yaml
```

To launch Muse Glimmer 30B LoRA training on a remote GCP 8x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/muse_glimmer/sft/30b_lora/gcp_job.yaml --cluster muse-glimmer-30b-lora
```

### Inference

```shell
oumi infer -i -c oumi://configs/recipes/muse_glimmer/inference/30b_infer.yaml
```

vLLM has no *native* `MuseGlimmerForConditionalGeneration` entry as of v0.27.0, so the
config uses Oumi's `NATIVE` engine. Add `adapter_model: <path>` under `model` to serve
a tuned LoRA adapter.

vLLM's transformers backend (`model_impl="transformers"`) **does not work** — tested
on vLLM 0.27.0, don't retry without an upstream fix:

- `tensor_parallel_size=2` fails outright: `mat1 and mat2 shapes cannot be multiplied
  (65536x768 and 1536x1536)` — the vision encoder's `hidden_size` (1536) gets sharded
  to 768 against an unsharded weight.
- `tensor_parallel_size=1` loads and serves, but emits garbage: chat prompts return a
  run of `<|eom|>` tokens, and a plain `"The capital of France is"` completion returns
  `'\n_conassistant\nThe'`. The generic backend does not reproduce this model's
  `final_logit_softcapping`, `output_multiplier`, `qk_scale_factor`, gated attention,
  or `MuseGlimmerTextCenteredRMSNorm`, so the logits are simply wrong.

LoRA serving on vLLM is therefore moot until a native `MuseGlimmer` implementation
lands. Note the native engine is slow: 200 rows took ~85 min on 4xH100.

## Validation

LoRA was validated on banking77 (77-way intent classification, single integer ID
per response), 200 held-out test rows, greedy decoding:

| | Exact match |
| :---- | :---: |
| Base | 0.605 |
| + LoRA (1 epoch, 104 steps) | **0.920** |

Training used the `text_completions_only_with_padding` collator with
`train_target: FINAL_ASSISTANT_TURN` so loss lands on the label rather than on the
77-line classifier prompt. Loss 0.577 → 0.149, token accuracy 0.89 → 0.96.

## Notes

- **Chat template.** Muse Glimmer's built-in template emits
  `<|start|>role<|message|>...<|eot|>` framing, and assistant turns render as
  `<|start|>assistant to=user<|message|>`. Oumi keeps the tokenizer's own template
  rather than substituting a bundled one.
- **Stop tokens are not optional.** Turns end on `<|eot|>` (`200008`), but the
  tokenizer's `eos_token` is `<|end_of_text|>` (`200001`), and Oumi's native engine
  defaults `stop_token_ids` to the tokenizer's `eos_token_id` alone — it does not
  read the model's `generation_config.json`, which lists both. With only `200001`,
  responses never terminate: the model answers and then keeps opening fresh
  `assistant to=user` turns until `max_new_tokens`. The inference config sets
  `stop_token_ids: [200001, 200008]`; keep it when writing eval configs.
- **Responses carry a ` to=user` prefix.** The generation prompt ends at
  `<|start|>assistant`, so the model itself emits the ` to=user<|message|>` recipient
  header and it lands in the decoded content. Strip it before exact-match scoring.
- **Text-only.** The recipes fine-tune the text transformer on text-only data. Oumi
  does not build a processor for this model, so image inputs are not supported for
  training.
