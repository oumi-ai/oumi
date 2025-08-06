# gpt-oss

## Summary

Configs for OpenAI's gpt-oss model family. See the [blog post](https://huggingface.co/blog/welcome-openai-gpt-oss) for more information. The models in this family, both of which are MoE architectures, are:

- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b): 21B total/3.6B active parameters
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): 117B total/5.1B active parameters

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.

## Example Commands

### Inference

To run interactive remote inference on gpt-oss-120b on Together AI:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_together_infer.yaml
```
