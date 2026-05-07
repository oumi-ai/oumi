# Gemma 4

## Summary

Configs for Google's Gemma 4 model family. See the [Hugging Face announcement](https://huggingface.co/blog/gemma4) and the [Google blog post](https://blog.google/technology/developers/gemma-4/) for more information. Models in this family include:

- Efficient (edge-targeted, multimodal: text + image + audio)
  - [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) (~5B)
  - [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (~8B) — **configs available**
- Larger (image + text, 256K context)
  - [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (MoE, 27B)
  - [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) (dense, 31B)

Gemma 4 requires accepting the model license on Hugging Face before downloading.

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
