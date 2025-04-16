# 🧀 Introducing HallOumi: a State-of-the-Art Claim Verification Model

[![Made with Oumi](https://badgen.net/badge/Made%20with/Oumi/%23085CFF?icon=https%3A%2F%2Foumi.ai%2Flogo_dark.svg)](https://github.com/oumi-ai/oumi)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/oumi-ai/HallOumi-8B)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg)](https://huggingface.co/collections/oumi-ai/halloumi-67ecccf60fa98a079ea41ea1)

We are excited to announce HallOumi, a truly open-source claim verification (hallucination detection) model, outperforming Claude Sonnet, OpenAI o1, DeepSeek R1, Llama 405B, and Gemini Pro at only 8B parameters. For more details, please see our [technical overview](https://oumi.ai/blog/posts/introducing-halloumi)!

| Model | Macro F1 Score | Balanced Accuracy | Open or Closed? | Model Size |
|-------|----------------|-------------------|----------------|------------|
| HallOumi 8B | 77.2% ± 2.2% | 74.8% ± 2.2% | Truly Open Source | 8B |
| Claude Sonnet 3.5 | 69.6% ± 2.8% | 67.3% ± 2.7% | Closed | Unknown |
| OpenAI o1-preview | 65.9% ± 2.3% | 64.5% ± 2.0% | Closed | Unknown |
| DeepSeek R1 | 61.6% ± 2.5% | 60.7% ± 2.1% | Open Weights | 671B |
| Llama 3.1 405B | 58.8% ± 2.4% | 58.7% ± 1.7% | Open Weights | 405B |
| Google Gemini 1.5 Pro | 48.2% ± 1.8% | 52.9% ± 1.0% | Closed | Unknown |
<br>

We are open-sourcing two versions of the HallOumi model:
- [HallOumi-8B](https://huggingface.co/oumi-ai/HallOumi-8B): A generative model that provides per-sentence classifications, breaks down claims into subclaims, and offers citations and rationale for its classifications.
- [HallOumi-8B-classifier](https://huggingface.co/oumi-ai/HallOumi-8B-classifier): A classifier model that operates at the claim level, delivering predictions along with confidence scores. This model is more computationally efficient, making it ideal for scenarios where compute resources and latency are a concern.

To try these out (without installation), please visit our [web demo](https://oumi.ai/halloumi-demo).

## 🛠 Setup

```bash
pip install oumi
```

## 🚀 Training

Example of Oumi fine-tuning:

```bash
# Train HallOumi-8B locally
oumi train -c oumi://configs/projects/halloumi/8b_train.yaml

# Launch a job to train HallOumi-8B on GCP
# Setup instructions: https://oumi.ai/docs/en/latest/user_guides/launch/launch.html
oumi launch up -c oumi://configs/projects/halloumi/gcp_job.yaml --cluster halloumi-8b-sft
```

## 🔄 Prompt Formatting

### HallOumi-8B

To construct a prompt for querying [HallOumi-8B](https://huggingface.co/oumi-ai/HallOumi-8B), you will need the following:

- `context` (`str`): A context document that serves as the premise (or ground truth).
- `request` (`str`): A request or question to a language model.
- `response` (`str`): An AI-generated or human-created response, which may consist of one or multiple claims (hypotheses). The objective is to ground each claim to the provided `context`.

For optimal efficiency, HallOumi requires the `context` and `response` to be broken down to sentences, and the prompt to be formatted as follows:

```
<|context|>
  <|s1|><1st sentence of the context><end||s>
  <|s2|><2nd sentence of the context><end||s>
  ...
<end||context>

<|request|>
  <The question asked to the LLM.>
<end||request>

<|response|>
  <|r1|><The 1st claim/sentence of the LLM response><end||r>
  <|r1|><The 2nd claim/sentence of the LLM response><end||r>
  ...
<end||response>
```

<!-- FIXME: HF prompt formatting does not seem up-to-date: https://huggingface.co/datasets/oumi-ai/oumi-groundedness-benchmark -->

Code sample: Please see the `create_prompt` helper function of our [inference notebook](https://github.com/oumi-ai/oumi/blob/main/configs/projects/halloumi/halloumi_inference_notebook.ipynb).

### HallOumi-8B-classifier

To construct a prompt for querying [HallOumi-8B-classifier](https://huggingface.co/oumi-ai/HallOumi-8B-classifier), you will need the following:

- `context` (`str`): A context document that serves as the premise (or ground truth).
- `claim` (`str`): A claim that must be validated or grounded in the context.

The prompt should be formatted as follows:

```
<context>
The context.
</context>

<claims>
A claim to be validated.
</claims>
```

Code sample: Please see our [classifier inference notebook](https://github.com/oumi-ai/oumi/blob/main/configs/projects/halloumi/halloumi_classifier_inference_notebook.ipynb).

## 🤖 Inference

### Local Inference

If you want to call our HallOumi API, or download HallOumi locally and run inference:
- [HallOumi-8B](https://huggingface.co/oumi-ai/HallOumi-8B): See our [inference notebook](https://github.com/oumi-ai/oumi/blob/main/configs/projects/halloumi/halloumi_inference_notebook.ipynb).
- [HallOumi-8B-classifier](https://huggingface.co/oumi-ai/HallOumi-8B-classifier): See our [inference notebook](https://github.com/oumi-ai/oumi/blob/main/configs/projects/halloumi/halloumi_classifier_inference_notebook.ipynb).

### Inference Server

You can easily host both models using SGLang:

#### Hosting HallOumi-8B (generative)
```shell
pip install sglang
python3 -m sglang.launch_server --model-path oumi-ai/HallOumi-8B --port 8000 --dtype auto --mem-fraction-static 0.9 --trust-remote-code
```

#### Hosting HallOumi-8B-classifier

```shell
pip install sglang
python3 -m sglang.launch_server --model-path oumi-ai/HallOumi-8B-classifier --port 8001 --dtype auto --mem-fraction-static 0.9 --trust-remote-code --is-embedding
```

Alternatively, you can self-host a version of our open-source web demo:
https://github.com/oumi-ai/halloumi-demo

## 📊 Evaluation

We have evaluated HallOumi-8B’s performance against multiple state-of-the-art models, including DeepSeek R1, OpenAI o1, Google Gemini 1.5 Pro, Llama 3.1 405B, and Claude Sonnet 3.5, using Oumi's [Groundedness Benchmark](https://huggingface.co/datasets/oumi-ai/oumi-groundedness-benchmark).

We are releasing a notebook that demonstrates how to run end-to-end comparative evaluations, using Oumi's custom evaluation framework. The notebook details how, given the set of prompts, you can run inference, extract model predictions from free-form text model responses, and calculate any relevant metrics you want (e.g., F1 and Balanced Accuracy) for any closed or open source model (including HallOumi).

Notebook for evaluation: [Evaluating LLMs as Hallucination Classifiers](https://github.com/oumi-ai/oumi/blob/main/configs/projects/halloumi/halloumi_eval_notebook.ipynb)

## 🏛️ License

This model is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

**NOTE**: This model was trained with the [ANLI subset](https://huggingface.co/datasets/oumi-ai/oumi-anli-subset), which requires the [Creative Commons NonCommercial (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license. You can [retrain the model](https://github.com/oumi-ai/oumi/blob/main/configs/projects/halloumi/8b_train.yaml) without this subset to avoid this restriction.



## 📖 Citation

If you use **HallOumi** in your research, please cite:

```
@misc{oumi2025HallOumi,
      title={HallOumi - a state-of-the-art claim verification model},
      author={Jeremiah Greer and Panos Achlioptas and Konstantinos Aisopos and Michael Schuler and Matthew Persons and Oussama Elachqar and Emmanouil Koukoumidis},
      year={2025},
      url={https://oumi.ai/halloumi},
}
```
