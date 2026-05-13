# Understanding Oumi: A Plain-English Guide

*This document explains the Oumi codebase for someone who has never written code. No technical background needed.*

---

## What is Oumi?

Imagine you want to teach a robot to be really good at answering questions, writing text, or understanding images. To do that, you need to:

1. Gather and organize a lot of example data (like textbooks for a student)
2. Run a "training" process (like letting the student study)
3. Test the student to see how well they learned
4. Deploy the trained student so others can use it

**Oumi is the complete toolkit that handles all of these steps.** It is a software platform that lets researchers and engineers build, train, test, and deploy AI language models — the same kind of AI that powers chatbots like ChatGPT or Claude.

Think of Oumi as a fully-equipped kitchen for cooking AI. Instead of building the oven, knives, and pots yourself, Oumi gives you all the tools in one place, ready to use.

---

## Who Uses It and Why?

| Who | Why They Use Oumi |
|-----|------------------|
| AI researchers | Run experiments without rebuilding everything from scratch |
| Companies | Train AI models on their own private data |
| Students | Learn how AI training works with real, production-grade tools |
| Open-source developers | Contribute to and improve the platform |

Oumi supports models ranging from small ones (that run on a laptop) to massive ones with 405 billion parameters (that require hundreds of specialized computers). It runs on personal machines, on-premise servers, and in the cloud (Amazon Web Services, Google Cloud, Microsoft Azure, and others).

---

## The Big Picture: What Happens When You Use Oumi?

Here is the journey of building an AI model with Oumi, step by step:

```
RAW DATA  →  TRAINING  →  TESTING  →  DEPLOYMENT  →  USERS
 (Step 1)     (Step 2)    (Step 3)     (Step 4)      (Step 5)
```

Each step has a dedicated part of the Oumi codebase. Let's walk through them.

---

## Step 1: Data — Teaching Material for the AI

**Folder:** `src/oumi/datasets/`

Before an AI can learn anything, it needs examples. These examples are called **datasets**. Think of datasets like textbooks:

- A dataset for question-answering looks like: *"Question: What is the capital of France? Answer: Paris."*
- A dataset for conversation looks like: *back-and-forth dialogue between a user and an assistant.*
- A dataset for image understanding looks like: *a picture paired with a description.*

### What the code does:
The `datasets/` folder contains code that:
- **Reads** data from many sources (files on your computer, or online repositories like HuggingFace Hub — a website that hosts thousands of AI datasets)
- **Formats** it into the exact shape the AI training process expects
- **Handles images and text together** for models that can see pictures

There are several types of datasets:
- **SFT (Supervised Fine-Tuning) datasets** — Pairs of instructions and ideal responses, used to teach the AI how to respond helpfully
- **Preference datasets** — Pairs of good and bad responses, used to teach the AI which answers are better
- **Vision-Language datasets** — Text combined with images, used to teach the AI to understand pictures

---

## Step 2: Training — The Learning Process

**Folder:** `src/oumi/core/trainers/` and `src/oumi/train.py`

Training is the process where the AI looks at thousands or millions of examples and adjusts itself to become better. Imagine a student who reads a flashcard, guesses the answer, checks if they're right, and then remembers for next time. AI training is essentially that, repeated billions of times.

### Types of Training Oumi Supports:

| Training Type | Simple Explanation |
|---------------|-------------------|
| **Full Fine-Tuning** | The AI updates every single "memory cell" it has — thorough but slow |
| **LoRA / QLoRA** | Only small "adapter" pieces are trained — faster and cheaper, almost as good |
| **DPO (Preference Tuning)** | Teach the AI to prefer good answers over bad ones |
| **GRPO** | Teach the AI to reason step-by-step, like solving math problems |
| **Distillation** | A large, smart AI teaches a smaller AI what it knows |

### Distributed Training:
Training a large AI requires enormous computing power. Oumi can spread the work across hundreds of computers working simultaneously (called **distributed training**). It supports industry-standard methods:
- **FSDP** — splits the AI's memory across many machines
- **DeepSpeed** — another way to efficiently split the work
- **DDP** — each machine trains on a different batch of data and shares what it learned

---

## Step 3: Evaluation — The Exam

**Folder:** `src/oumi/evaluation/` and `src/oumi/evaluate.py`

After training, you need to check: *How well does the AI actually perform?* Oumi runs the AI through standardized "exams" called benchmarks.

### Common Benchmarks:
| Benchmark | What It Tests |
|-----------|--------------|
| **MMLU** | General knowledge across 57 subjects |
| **GSM8K** | Grade-school math word problems |
| **HumanEval** | Writing working computer code |
| **TruthfulQA** | Avoiding making up false information |
| **HellaSwag** | Completing sentences sensibly |

Oumi automatically runs these tests and reports scores so you can compare different versions of your AI.

---

## Step 4: Inference — Using the Trained AI

**Folder:** `src/oumi/inference/`

Once training is done, you want to actually *use* the AI — ask it questions and get answers. This process is called **inference** (running the model to produce output).

Oumi has 24 different **inference engines** — different systems for running the AI depending on your needs:

| Engine | When to Use It |
|--------|---------------|
| **Native (PyTorch)** | Simple, general-purpose usage |
| **vLLM** | Fast, efficient serving for many users at once |
| **llama.cpp** | Running on a regular laptop (CPU, no special GPU needed) |
| **OpenAI API** | When you want to use OpenAI's ChatGPT in your pipeline |
| **Anthropic API** | When you want to use Claude AI |
| **Bedrock, Vertex AI** | Using AI services from Amazon or Google |

Think of these like different types of vehicles. A bicycle (llama.cpp) works fine for a short personal trip. A bus (vLLM) is better when many people need rides at once. A rental car service (OpenAI API) is useful when you don't want to maintain the vehicle yourself.

---

## Step 5: Deployment — Making the AI Available

**Folder:** `src/oumi/deploy/`

Deployment means putting the trained AI on a server so real users can access it. Oumi supports deploying to:
- **Fireworks.ai** — a service specialized in hosting AI models
- **Parasail** — another fast, low-delay AI hosting service

The deployment code handles: uploading the model, configuring it, and managing the connection between users and the AI.

---

## Supporting Systems

Beyond the main pipeline, Oumi has several supporting tools:

### Judges — The AI Quality Inspector
**Folder:** `src/oumi/judges/`

A "judge" is itself an AI that evaluates other AI outputs. Imagine using one expert teacher to grade papers written by student AIs. Judges are used to:
- Filter out bad training data before training begins
- Score the quality of AI-generated text
- Validate that synthetic data makes sense

### Data Synthesis — Creating Training Data from Scratch
**Folder:** `src/oumi/core/synthesis/`

Sometimes you don't have enough real data. Oumi can **synthesize** (artificially create) training data:
- It can read documents (PDFs, Word files, spreadsheets)
- Use a large AI to generate question-answer pairs based on those documents
- Create thousands of conversation examples automatically

### Job Launcher — Running Work on Cloud Computers
**Folder:** `src/oumi/launcher/`

Training large AI models requires renting powerful computers in the cloud. The launcher handles:
- Connecting to cloud providers (AWS, Google Cloud, Azure, Lambda Labs, etc.)
- Starting up computer clusters
- Submitting training jobs
- Monitoring progress
- Shutting everything down when done

This is like a travel agent that books your flight, hotel, and activities — you just say where you want to go.

### Analysis — Understanding Your Data
**Folder:** `src/oumi/analyze/`

Before training, it helps to understand what your data looks like: How long are the examples? Are there duplicates? Is the quality good? The analysis tools provide statistics and visualizations about your datasets.

### Quantization — Making the AI Smaller
**Folder:** `src/oumi/quantize/`

A trained AI can be very large (hundreds of gigabytes). Quantization is a compression technique that makes the AI smaller and faster with minimal quality loss — similar to how a JPEG image takes less storage than a RAW photo while still looking good.

---

## Configuration — The Recipe Card System

**Folder:** `configs/`

Every action in Oumi is controlled by a **configuration file** — a text file that lists all the settings. Think of it like a recipe card:

```
Recipe: Train Qwen3-8B with ShareGPT data

Ingredients:
  Model: Qwen/Qwen3-8B
  Data: ShareGPT3K dataset
  Training time: 3 passes through all data
  Learning rate: 0.0002
  Technique: LoRA (efficient fine-tuning)
```

Oumi comes with **200+ pre-written recipes** for popular AI models. Instead of figuring out all the settings yourself, you can start from a proven recipe and adjust it.

Models with pre-built recipes include:
- **Llama** (Meta's open AI models)
- **Qwen** (Alibaba's models)
- **DeepSeek** (specialized in reasoning and coding)
- **Phi** (Microsoft's small but capable models)
- **Gemma** (Google's models)
- And 35+ more

---

## The Command Line Interface (CLI) — How Humans Tell Oumi What to Do

**Folder:** `src/oumi/cli/`

Users interact with Oumi by typing commands into a terminal (the text-based computer interface). Here are the main commands:

| Command | What It Does |
|---------|-------------|
| `oumi train` | Start training an AI model |
| `oumi evaluate` | Run the AI through benchmark exams |
| `oumi infer` | Chat with or use the trained AI |
| `oumi judge` | Score a dataset for quality |
| `oumi synth` | Synthesize new training data |
| `oumi launch` | Send a training job to a cloud cluster |
| `oumi deploy` | Deploy the model to a serving endpoint |
| `oumi analyze` | Analyze a dataset's statistics |

---

## Notebooks — Interactive Learning Guides

**Folder:** `notebooks/`

Oumi provides 15+ **Jupyter notebooks** — interactive documents that mix explanations, code, and live results. Think of them like hands-on lab workbooks where each step can be run and the output appears immediately below.

Examples include:
- *A Tour of Oumi* — a quick introduction
- *Finetuning Tutorial* — full walkthrough of customizing an AI
- *Model Evaluation* — how to measure AI performance
- *Vision Language Models* — using AI that understands images
- *Running Jobs Remotely* — cloud-based training walkthrough

---

## How the Codebase Is Organized

Here's a map of the main folders and what they contain:

```
oumi/
│
├── src/oumi/              ← All the working code lives here
│   ├── core/              ← The engine room: trainers, configs, datasets base classes
│   ├── datasets/          ← Specific dataset implementations
│   ├── inference/         ← 24 ways to run an AI model
│   ├── judges/            ← AI-evaluates-AI systems
│   ├── evaluation/        ← Benchmark testing
│   ├── deploy/            ← Hosting the AI online
│   ├── launcher/          ← Cloud job management
│   ├── mcp/               ← Integration with AI coding assistants (Claude, Cursor)
│   ├── analyze/           ← Data statistics and inspection
│   ├── quantize/          ← Model compression
│   ├── cli/               ← Command-line interface
│   ├── builders/          ← Factory code that creates components
│   └── utils/             ← Small helper functions used everywhere
│
├── configs/               ← 200+ pre-built YAML recipe files
│   └── recipes/           ← Organized by model and task
│
├── notebooks/             ← 15+ interactive tutorial notebooks
│
├── scripts/               ← One-off utility scripts
│
├── docs/                  ← Written documentation (Sphinx format)
│
├── tests/                 ← Automated tests to ensure the code works
│
├── Dockerfile             ← Instructions for running Oumi in a container
│
└── pyproject.toml         ← Project metadata and dependency list
```

---

## Key Concepts Explained Simply

### What is a "model" or "parameters"?
An AI model is a mathematical structure with billions of adjustable numbers ("parameters"). Training adjusts those numbers until the model's outputs match what we want. More parameters = more capable (but slower and more expensive) model.

### What is "fine-tuning"?
Pre-trained AI models are trained on massive, general internet data. Fine-tuning means taking that pre-trained model and teaching it to be better at a *specific* task — like customer support, medical advice, or coding.

### What is "HuggingFace"?
HuggingFace is a popular website and platform (like GitHub, but for AI models and datasets). Oumi integrates deeply with it — you can download models and datasets from HuggingFace with a single line of configuration.

### What is "GPU"?
A GPU (Graphics Processing Unit) is specialized computer hardware originally designed for video games. AI training requires millions of identical math operations in parallel — exactly what GPUs excel at. Large models need many GPUs working together.

### What is "YAML"?
YAML is a human-readable text format for writing configuration. It looks like a structured list of settings. Oumi's configuration files are all written in YAML.

### What is "open source"?
Oumi is licensed under Apache 2.0, which means anyone can read the code, use it for free, modify it, and contribute improvements back. There is no vendor to pay or lock-in.

---

## What Makes Oumi Special?

Most AI tools solve one part of the pipeline (just training, or just inference, or just evaluation). Oumi does **all of it** with a single consistent system. The same configuration format, the same CLI commands, and the same codebase handle everything from raw data to a live deployed service.

Key strengths:
- **No vendor lock-in** — works with 40+ models and 24 inference engines
- **Scales from laptop to cluster** — the same code runs on a MacBook or 1,000 GPUs
- **Research-grade flexibility** — customize every component for experiments
- **Production-grade reliability** — checkpoint saving, error recovery, logging
- **Open source community** — 10,000+ GitHub stars, active Discord

---

## Summary

Oumi is like a factory assembly line for building AI models:

1. **Raw materials come in** (datasets — text, images, conversations)
2. **The factory processes them** (training — the AI learns from examples)
3. **Quality control checks the product** (evaluation — running benchmarks)
4. **The product is packaged** (inference engines — the AI is made usable)
5. **The product ships to customers** (deployment — the AI is hosted online)

Every machine in the factory (every part of the codebase) is designed to work smoothly with the others, and you can swap out individual machines without rebuilding everything else.

The result is that a small team — or even one person — can go from a blank slate to a custom AI model running in production, using battle-tested tools without needing to build infrastructure from scratch.

---

*Document generated May 2026. Oumi is open source under the Apache 2.0 license.*
