# Inference Engines

## Overview
Inference Engines are simple tools for running inference on models in Oumi. This includes newly trained models, downloaded pretrained models, and even remote APIs such as Anthropic, Gemini, and Open AI.

## Choosing an Engine

Our engines are broken into two categories: local inference vs remote inference. But how do you decide between the two?

Generally, the answer is simple: if you can enough resources to run the model locally without OOMing, then use a local engine like {py:obj}`~oumi.inference.VLLMInferenceEngine`, {py:obj}`~oumi.inference.NativeTextInferenceEngine`, or {py:obj}`~oumi.inference.LlamaCppInferenceEngine`.

If you don't have enough local compute resources, then the model must be hosted elsewhere. Our remote inference engines assume that your model is hosted behind a remote API. You can use {py:obj}`~oumi.inference.AnthropicInferenceEngine`, or {py:obj}`~oumi.inference.GoogleVertexInferenceEngine` to call their respective APIs. You can also use {py:obj}`~oumi.inference.RemoteInferenceEngine` to call any API implementing the Open AI Chat API format (including Open AI's native API).

For a comprehensive list of engines, see the [Supported Engines](#supported-engines) section below.

## Loading an Engine

Now that you've decided on the engine you'd like to use, you'll need to create a small config to instantiate your engine.

All engines require a model, specified via {py:obj}`~oumi.core.configs.ModelParams`. Any engine calling an external API / service (such as Anthropic, Gemini, Open AI, or a self-hosted server) will also require {py:obj}`~oumi.core.configs.RemoteParams`.

See {py:obj}`~oumi.inference.NativeTextInferenceEngine` for an example of a local inference engine and {py:obj}`~oumi.inference.AnthropicInferenceEngine` for an example of an inference engine that requires a remote API.

## Supported Engines

```{include} ../../api/summary/inference_engines.md
```

## Inference Engine Configs

```{eval-rst}
.. literalinclude:: ../../../src/oumi/core/configs/params/generation_params.py
    :language: python
    :caption: GenerationParams
    :pyobject: GenerationParams
```
