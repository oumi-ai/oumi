# Gemma 4

## Summary

Configs for Google's Gemma 4 model family. See the [Hugging Face announcement](https://huggingface.co/blog/gemma4) and the [Google blog post](https://blog.google/technology/developers/gemma-4/) for more information. Models in this family include:

- Efficient (edge-targeted, multimodal: text + image + audio)
  - [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) (~5B) — **FFT + LoRA configs available**
  - [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (~8B) — **FFT + LoRA configs available**
- Unified (encoder-free, multimodal: text + image + audio)
  - [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) (12B) — **FFT + LoRA configs available**
- Larger (image + text, 256K context)
  - [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (MoE, 26B) — **LoRA config available**
  - [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) (dense, 31B) — **LoRA config available**

Gemma 4 requires accepting the model license on Hugging Face before downloading.
Training requires `transformers >= 5.5.4` for the E2B/E4B/31B recipes (installed
automatically by oumi) and `transformers >= 5.10.0` for 12B, which must be upgraded
manually (`uv pip install -U "transformers>=5.10"`) because oumi currently pins
`transformers<5.10`.

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) if you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.
4. (Optional) If you wish to do deeper experimentation, follow our [instructions](https://oumi.ai/docs/en/latest/development/dev_setup.html) to clone the Oumi repository locally.
   - Make sure to delete the `oumi://` prefix when running Oumi commands, to disable fetching the latest configs from GitHub!

## Example Commands

### Training

To launch Gemma 4 E4B FFT training locally:

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/gemma4/sft/e4b_full/train.yaml
```

To launch Gemma 4 E4B FFT training on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/e4b_full/gcp_job.yaml --cluster gemma4-e4b-full
```

To launch Gemma 4 E2B FFT training locally with FSDP:

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/gemma4/sft/e2b_full/train.yaml
```

To launch Gemma 4 E2B FFT training on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/e2b_full/gcp_job.yaml --cluster gemma4-e2b-full
```

To launch Gemma 4 12B FFT training locally with FSDP (needs a multi-GPU node):

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/gemma4/sft/12b_full/train.yaml
```

To launch Gemma 4 12B FFT training on a remote GCP 8x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/12b_full/gcp_job.yaml --cluster gemma4-12b-full
```

### LoRA Training

LoRA is scoped to the language-model layers only. Gemma 4's non-text towers use
`Gemma4ClippableLinear` wrappers that PEFT cannot adapt, and they share projection
names (`q_proj`, `v_proj`, ...) with the text model. The recipes target the plain
projection names and set `lora_exclude_modules` to keep LoRA off the towers: the
Efficient (text+image+audio) models exclude `[".*vision_tower.*", ".*audio_tower.*"]`,
and the Larger (image+text) models exclude `[".*vision_tower.*", ".*multi_modal_projector.*"]`.
The Unified 12B is encoder-free — image/audio enter through `embed_vision`/`embed_audio`
modules rather than named towers — so it excludes `[".*vision.*", ".*audio.*"]`.
oumi passes this list to PEFT's `exclude_modules`.

To launch Gemma 4 E4B LoRA training locally (fits a single A100/H100):

```shell
oumi train -c oumi://configs/recipes/gemma4/sft/e4b_lora/train.yaml
```

To launch Gemma 4 12B LoRA training locally with FSDP:

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/gemma4/sft/12b_lora/train.yaml
```

To launch Gemma 4 E2B LoRA training locally (fits a single 32GB GPU):

```shell
oumi train -c oumi://configs/recipes/gemma4/sft/e2b_lora/train.yaml
```

To launch Gemma 4 E4B LoRA training on a remote GCP A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/e4b_lora/gcp_job.yaml --cluster gemma4-e4b-lora
```

To launch Gemma 4 26B (MoE) LoRA training on a remote GCP 8x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/26b_lora/gcp_job.yaml --cluster gemma4-26b-lora
```

To launch Gemma 4 31B LoRA training on a remote GCP 8x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/31b_lora/gcp_job.yaml --cluster gemma4-31b-lora
```

To launch Gemma 4 12B LoRA training on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/gemma4/sft/12b_lora/gcp_job.yaml --cluster gemma4-12b-lora
```
