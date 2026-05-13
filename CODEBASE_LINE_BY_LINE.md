# Oumi Codebase — Line-by-Line Guide to the Most Important Files

*This document walks through the most central source files in the Oumi codebase.
Every significant line is explained in plain English — no coding knowledge assumed.*

---

## Table of Contents

1. [The Front Door: `__init__.py`](#1-the-front-door-__init__py)
2. [Training an AI: `train.py`](#2-training-an-ai-trainpy)
3. [Using the AI: `infer.py`](#3-using-the-ai-inferpy)
4. [Testing the AI: `evaluate.py`](#4-testing-the-ai-evaluatepy)
5. [Error Vocabulary: `exceptions.py`](#5-error-vocabulary-exceptionspy)
6. [The Recipe Card: `training_config.py`](#6-the-recipe-card-training_configpy)
7. [Trainer Choices: `training_params.py` (TrainerType)](#7-trainer-choices-training_paramspy)
8. [The Message Format: `conversation.py`](#8-the-message-format-conversationpy)
9. [The Factory Hub: `builders/__init__.py`](#9-the-factory-hub-builders__init__py)
10. [Building Models: `builders/models.py`](#10-building-models-buildersmdelspy)
11. [Loading Datasets: `builders/data.py`](#11-loading-datasets-buildersdatapy)

---

## 1. The Front Door: `__init__.py`

**File:** `src/oumi/__init__.py`

This is the very first file Python reads when you write `import oumi`. Think of it as the lobby of a building — it doesn't do the work itself, but it tells you which rooms contain what.

```
Line 15-16:  """Oumi (Open Universal Machine Intelligence) library.
```
This is a docstring — a text description. It says: "this file is the entry point for the Oumi library."

```
Lines 84-108  (imports and TYPE_CHECKING)
```
Python has to load code from other files before it can use it. These lines do that loading. `TYPE_CHECKING` is a special trick: it means "only load these imports when someone is reading the code for documentation purposes, not when actually running it" — this makes startup faster.

```python
logging.configure_dependency_warnings()
```
Line 108: As soon as anyone imports Oumi, this line runs immediately. It sets up the logging system (the mechanism that prints status messages to the screen) and silences noisy warnings from third-party libraries.

---

### The Public Functions

Each function below is a thin "pass-through" — it accepts your instructions, then calls the real implementation in a separate file. This is a deliberate design: the front door stays simple, the complexity lives deeper inside.

```python
def evaluate_async(config: AsyncEvaluationConfig) -> None:
```
**Line 111:** Defines a function called `evaluate_async`. The `config` is the recipe card you hand it. The `-> None` means it doesn't return anything (it just runs). This function lets you evaluate a model *while it's still training* — useful for seeing progress in real time.

```python
    import oumi.evaluate_async
    return oumi.evaluate_async.evaluate_async(config)
```
Lines 126-128: The function body. It loads the real evaluation module and immediately calls the real function inside it. The import is *inside* the function (not at the top of the file) — this is intentional. Python doesn't load the heavy AI libraries until you actually call this function, making `import oumi` fast.

---

```python
def evaluate(config: EvaluationConfig) -> list[dict[str, Any]]:
```
**Line 131:** Defines the simpler synchronous evaluate function. The `-> list[dict[str, Any]]` part means it returns a *list of dictionaries* — imagine a table where each row is one benchmark result and each column is a metric name and score.

---

```python
def infer_interactive(...) -> None:
```
**Line 146:** The function for interactive chatting with a model. You type a message, the AI responds, repeat. Takes optional `input_image_bytes` (raw image data if you want to show the AI a picture), `system_prompt` (instructions like "You are a helpful assistant"), and `console` (for displaying a spinning loading indicator).

---

```python
def infer(config, inputs, inference_engine, *, input_image_bytes) -> list[Conversation]:
```
**Line 173:** Batch inference — send multiple questions at once, get multiple answers back. Returns a `list[Conversation]` — a list of complete dialogues (each dialogue contains the question and the AI's response).

---

```python
def judge_dataset(judge_config, dataset) -> list[JudgeOutput]:
```
**Line 200:** Runs every item in a dataset through a "judge" AI that scores quality. Returns `list[JudgeOutput]` — one quality score per dataset item.

---

```python
def synthesize(config: SynthesisConfig) -> list[dict[str, Any]]:
```
**Line 254:** Creates artificial training data using an AI. Returns the newly generated data as a list of dictionaries (each dictionary is one data example).

---

```python
def train(config: TrainingConfig, ...) -> dict[str, Any] | None:
```
**Line 261:** Launches the full training process. Returns either `None` (if just training) or a dictionary of final evaluation metrics (if also tuning hyperparameters).

---

```python
def quantize(config: QuantizationConfig) -> QuantizationResult:
```
**Line 278:** Compresses a model to make it smaller and faster. Returns a `QuantizationResult` with the path to the compressed model and its new size.

---

```python
__all__ = ["evaluate_async", "evaluate", "infer_interactive", ...]
```
**Lines 309-318:** This list tells Python exactly which names are available when someone writes `from oumi import *`. It's the official public menu of the library.

---

## 2. Training an AI: `train.py`

**File:** `src/oumi/train.py`

This is the most important file in Oumi. It orchestrates the entire training process — think of it as the "conductor" of an orchestra who coordinates all the musicians (model, data, GPU, distributed computing) to play together.

---

### The Imports (Lines 15–84)

```python
import functools
import time
```
**Lines 15-16:** Standard Python tools. `functools` allows wrapping functions for later use. `time` lets the code measure how long things take.

```python
from collections.abc import Callable
from pathlib import Path
from pprint import pformat
from typing import Any, Final, cast
```
**Lines 17-20:** More standard tools. `Callable` means "a function that can be called." `Path` represents file/folder locations. `pformat` creates human-readable text from complex objects (for logging). `Final` means a variable that will never change. `cast` tells the type-checker to treat a variable as a specific type.

```python
import datasets as hf_datasets
import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint
```
**Lines 22-25:** The heavy AI libraries. `datasets` is HuggingFace's dataset library (renamed `hf_datasets` to avoid name collisions). `torch` is PyTorch — the mathematical engine that runs the AI. `transformers` is HuggingFace's library of pre-built AI models. `get_last_checkpoint` is a utility to find the most recent saved training snapshot.

```python
from oumi.builders import (
    build_collator_from_config,
    build_dataset_mixture,
    ...
)
```
**Lines 27-40:** Imports Oumi's own "factory functions" — functions that create the individual pieces needed for training (dataset, model, trainer, etc.).

```python
from oumi.core.configs import (
    DatasetSplit,
    TrainerType,
    TrainingConfig,
)
```
**Lines 41-45:** Imports the configuration classes. `DatasetSplit` is an enum (a fixed set of choices) for TRAIN/VALIDATION/TEST. `TrainerType` is the list of supported trainer algorithms. `TrainingConfig` is the master recipe card.

```python
from oumi.core.distributed import (
    barrier,
    cleanup_distributed,
    ...
    init_distributed,
    is_distributed,
    is_local_process_zero,
    is_world_process_zero,
    ...
)
```
**Lines 50-61:** Imports tools for distributed training (spreading work across many computers). `barrier()` makes all computers pause and wait for each other to reach the same point — like everyone counting "1, 2, 3" before starting a race together. `is_world_process_zero()` returns True only on the "head" computer (the one responsible for saving files and printing logs).

---

### Helper Functions (Lines 87–239)

```python
def _find_checkpoint_to_resume_from(
    resume_from_checkpoint: str | None,
    try_resume_from_last_checkpoint: bool,
    output_dir: str,
) -> str | None:
```
**Lines 87-91:** A helper function (note the underscore prefix `_` — that means "for internal use only"). It finds the path to a previously saved training snapshot so training can continue from where it left off. Returns the path as a string, or `None` if no checkpoint is found.

```python
    checkpoint_path = None
    if resume_from_checkpoint:
        checkpoint_path = resume_from_checkpoint
    elif try_resume_from_last_checkpoint:
        checkpoint_path = get_last_checkpoint(output_dir)
        if not checkpoint_path:
            logger.warning(f"No checkpoints found under {output_dir}")
```
**Lines 94-99:** Logic: if you gave an explicit path, use it; otherwise search the output directory for the latest snapshot. If none is found but you asked for one, print a warning.

---

```python
def _ensure_dir_exists(output_dir: str | Path, human_readable_name: str) -> None:
```
**Line 107:** Makes sure a folder exists before trying to save files into it. `human_readable_name` is just a friendly label used in error messages (e.g., "training output directory").

```python
    output_dir_path: Path = Path(output_dir)
    if output_dir_path.exists():
        if not output_dir_path.is_dir():
            raise ValueError(...)
    elif is_local_process_zero():
        output_dir_path.mkdir(parents=True, exist_ok=True)
```
**Lines 110-118:** Converts the path to a `Path` object, checks if it already exists, and if not creates it. `parents=True` means "also create all parent folders if needed." Only the main process (not worker processes) creates the directory — `is_local_process_zero()` prevents multiple GPUs from fighting to create the same folder.

---

```python
def _finalize_training_config(config: TrainingConfig) -> TrainingConfig:
```
**Line 159:** Resolves any settings that couldn't be determined until runtime. For example, the number of data-loading worker threads can be set to `"auto"` — this function replaces that placeholder with the actual optimal number.

```python
    if config.training.dataloader_num_workers == "auto":
        num_workers = estimate_dataloader_num_workers()
```
**Lines 161-163:** If the user wrote `"auto"`, call a function that figures out the right number based on available CPU cores.

```python
    if config.training.trainer_type == TrainerType.TRL_GRPO:
        ...
        if num_generations is not None and global_batch_size % num_generations != 0:
            logger.warning(...)
```
**Lines 172-184:** Special validation for GRPO training: the number of responses generated per question must divide evenly into the total batch size (otherwise the math doesn't work out). Prints a warning if it doesn't.

---

```python
def _create_optional_training_kwargs(
    config, trainer_type, metrics_function, reward_functions, ...
) -> dict[str, Any]:
```
**Line 189:** Builds the dictionary of keyword arguments to pass to the trainer. Different trainer types need different arguments — this function figures out which arguments apply to which trainer.

```python
    if trainer_type == TrainerType.TRL_GRPO:
        if metrics_function:
            raise ValueError(f"metrics_function isn't supported for {trainer_type}")
```
**Lines 205-207:** GRPO trainers don't support custom metric functions (they have their own reward system). If someone accidentally passes both, this raises a clear error rather than failing silently.

---

### The VERL Training Path (Lines 242–278)

```python
def _verl_train(partial_trainer: Callable[[], BaseTrainer]):
```
**Line 242:** A special function for "VERL GRPO" training — a very advanced distributed training method that uses a framework called Ray.

```python
    import ray
```
**Line 249:** Imports Ray only if this function is called (lazy import). Ray is a framework for distributing Python code across many machines.

```python
    ray.init(runtime_env={"env_vars": {...}})
```
**Lines 256-263:** Starts a Ray cluster (network of computing nodes). Sets environment variables that control how GPUs communicate and how verbose logging is.

```python
    @ray.remote
    def _run_verl_train(partial_trainer):
        trainer = partial_trainer()
        trainer.train()
```
**Lines 269-273:** Defines a function that runs *remotely* on a Ray worker. `@ray.remote` is a decorator — it transforms the function so it can be sent to another machine to run.

---

### The Main `train()` Function (Lines 280–597)

This is the heart of the entire codebase.

```python
def train(
    config: TrainingConfig,
    additional_model_kwargs: dict[str, Any] | None = None,
    additional_trainer_kwargs: dict[str, Any] | None = None,
    additional_tuning_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> None | dict[str, Any]:
    """Trains a model using the provided configuration."""
```
**Lines 280-287:** The function signature. `config` is the full recipe card. The `additional_*_kwargs` allow passing extra settings beyond what the config supports. `verbose=False` means "don't print the full config unless asked."

```python
    _START_TIME = time.time()
```
**Line 288:** Records the exact moment training begins (as a Unix timestamp in seconds) so we can measure total training time later.

```python
    _create_training_dirs(config)
    _log_training_info(config)
```
**Lines 290-291:** Sets up folders and prints version/device information before anything else.

```python
    log_dir = Path(config.training.output_dir) / "logs"
    for logger_name in ("oumi", "oumi.telemetry"):
        configure_logger(logger_name, level=config.training.log_level, log_dir=log_dir)
```
**Lines 294-296:** Creates a `logs/` subfolder inside the output directory and configures two loggers to write messages there. One for regular output, one for performance/telemetry data.

```python
    if (
        config.deepspeed
        and config.deepspeed.is_zero3_enabled()
        and config.deepspeed.stage3_gather_16bit_weights_on_model_save
        and get_device_rank_info().world_size > get_device_rank_info().local_world_size
    ):
        logger.warning("⚠️  Multi-node DeepSpeed ZeRO-3 model saving detected...")
```
**Lines 302-316:** A safety check. When using the most aggressive form of DeepSpeed distributed training across multiple physical machines, saving model weights can cause the program to freeze. This prints a prominent warning if that risky combination is detected.

```python
    tokenizer: BaseTokenizer | None = None
    if is_custom_model(config.model.model_name) and not config.model.tokenizer_name:
        tokenizer = None
    else:
        tokenizer = build_tokenizer(config.model)
```
**Lines 327-332:** Creates the tokenizer — the component that converts human text (like "Hello") into numbers (like `[15339]`) that the AI can process. Custom (non-standard) models may not need a separate tokenizer, so it can be `None`.

```python
    processor: BaseProcessor | None = None
    if is_image_text_llm(config.model):
        processor = build_processor(...)
```
**Lines 334-345:** For vision-language models (AIs that understand both text and images), a "processor" is needed in addition to the tokenizer. It handles image resizing and encoding. `is_image_text_llm` checks if the configured model works with images.

```python
    train_dataset = build_dataset_mixture(
        config.data, tokenizer, DatasetSplit.TRAIN, seq_length=...
    )
```
**Lines 360-365:** Loads and prepares the training data. `DatasetSplit.TRAIN` selects the training portion of the dataset (as opposed to the validation portion used for checking progress).

```python
    eval_dataset = None
    if len(config.data.get_split(DatasetSplit.VALIDATION).datasets) != 0:
        eval_dataset = build_dataset_mixture(...)
```
**Lines 367-374:** Loads the validation dataset — the held-out data used to measure progress without cheating (the AI never trains directly on this data). Only loaded if validation data is actually specified in the config.

```python
    trainer_type: Final[TrainerType] = config.training.trainer_type
    metrics_function: Callable | None = build_metrics_function(config.training)
    reward_functions: list[Callable] = build_reward_functions(config.training)
    rollout_function: Callable | None = build_rollout_function(config.training)
```
**Lines 376-379:** Reads what type of trainer to use and builds the optional measurement functions. `metrics_function` calculates accuracy/scores during evaluation. `reward_functions` score responses during GRPO reinforcement training. `rollout_function` controls how new examples are generated during training.

```python
    if trainer_type == TrainerType.TRL_GRPO:
        if len(reward_functions) == 0:
            logger.warning(f"No reward_function specified for {trainer_type}!")
```
**Lines 380-382:** GRPO training is useless without reward functions (there's nothing to optimize toward). This prints a warning if someone forgot to configure them.

```python
    if config.training.trainer_type == TrainerType.VERL_GRPO:
        ...
        _verl_train(partial_trainer)
        return
```
**Lines 413-430:** If using the VERL variant of GRPO, hand off to the special `_verl_train` function (which uses Ray) and return early — the rest of this function doesn't apply.

```python
    checkpoint_location = _find_checkpoint_to_resume_from(...)
```
**Line 432:** Determines if there's a saved checkpoint to resume from.

```python
    if is_distributed():
        init_distributed(timeout_minutes=config.training.nccl_default_timeout_minutes)
```
**Lines 438-439:** If running across multiple GPUs/machines, initialize the communication layer between them. `NCCL` (NVIDIA Collective Communications Library) is the software that lets GPUs talk to each other. The timeout prevents the program from hanging forever if a GPU goes silent.

```python
    use_peft = config.training.use_peft and config.peft
    model = build_model(
        model_params=config.model,
        peft_params=config.peft if use_peft else None,
        ...
    )
```
**Lines 459-466:** Finally loads the actual AI model into memory. `use_peft` is True if using efficient fine-tuning (LoRA/QLoRA) rather than updating all parameters.

```python
    if use_peft:
        model = build_peft_model(model, config.training.enable_gradient_checkpointing, config.peft)
```
**Lines 468-470:** If using efficient fine-tuning, wraps the model with the PEFT adapter layer (the small trainable add-on that sits on top of the frozen base model).

```python
    if is_local_process_zero():
        log_number_of_model_parameters(model)
```
**Lines 474-475:** Prints how many trainable parameters the model has (e.g., "7.24B parameters"). Only the main process prints this to avoid duplicate output from every GPU.

```python
    device_cleanup()
```
**Line 513:** Frees unused GPU memory before training starts — like clearing your desk before starting a big project.

```python
    with torch_profile(...) as profiler:
```
**Line 515:** Opens a performance profiler. Everything inside this `with` block will be timed and measured, producing a report of where time was spent.

```python
        callbacks = build_training_callbacks(config, model, profiler)
        trainer = create_trainer_fn(
            model=model,
            processing_class=tokenizer,
            args=config.training,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            **training_kwargs,
        )
```
**Lines 521-531:** Builds the trainer object by assembling all the pieces together: model, tokenizer, training settings, datasets, and callbacks (callbacks are functions that run at specific moments during training, like "save checkpoint every 500 steps").

```python
        barrier()
```
**Line 561:** All GPUs/machines must reach this line before any of them start training. This synchronization prevents faster machines from beginning while others are still loading data.

```python
        logger.info(f"Training init time: {time.time() - _START_TIME:.3f}s")
        trainer.train(resume_from_checkpoint=checkpoint_location)
```
**Lines 564-570:** Logs setup time and then calls `trainer.train()` — the single line that actually starts the learning process. Everything before this was setup; this is where the AI actually studies.

```python
    logger.info("Training is Complete.")
    log_peak_gpu_memory()
```
**Lines 572-575:** After training finishes, prints a completion message and how much GPU memory was used at peak (useful for capacity planning).

```python
    if config.training.save_final_model:
        trainer.save_state()
        barrier()
        trainer.save_model(config=config)
```
**Lines 578-586:** Saves the trained model to disk. `save_state()` saves optimizer state and training metadata. `save_model()` saves the actual AI weights. `barrier()` between them ensures all GPUs have finished before saving begins.

```python
    if is_distributed():
        cleanup_distributed()
```
**Lines 590-591:** Shuts down the GPU-to-GPU communication network cleanly.

```python
    if additional_tuning_kwargs:
        return {**trainer.get_last_eval_metrics()}
    _log_feedback_request()
```
**Lines 593-597:** If called as part of hyperparameter tuning, returns the final metrics so the tuner can compare this run to others. Otherwise, prints a friendly message asking for user feedback.

---

## 3. Using the AI: `infer.py`

**File:** `src/oumi/infer.py`

This file handles *using* a trained AI — sending it messages and getting responses.

---

```python
def get_engine(config: InferenceConfig) -> BaseInferenceEngine:
```
**Line 33:** A helper that reads the config and creates the right inference engine. If no engine is specified, defaults to the "native" (built-in PyTorch) engine.

```python
    return build_inference_engine(
        engine_type=config.engine or InferenceEngineType.NATIVE,
        model_params=config.model,
        remote_params=config.remote_params,
    )
```
**Lines 39-43:** `config.engine or InferenceEngineType.NATIVE` means "use whatever engine the user specified, or use NATIVE if nothing was specified." `remote_params` contains API keys and endpoint URLs for cloud inference services.

---

```python
def infer_interactive(config, *, input_image_bytes, system_prompt, console) -> None:
```
**Line 46:** The interactive chat function. The `*` before the keyword arguments means they *must* be named explicitly when called (e.g., `system_prompt="..."`, not just `"..."`).

```python
    inference_engine = get_engine(config)
    while True:
```
**Lines 64-65:** Creates the inference engine *once* before the loop (not inside the loop) — loading an AI model takes time, so we only do it once and reuse it for every message.

`while True:` creates an infinite loop. The only way to exit is via the `except (EOFError, KeyboardInterrupt)` handler below.

```python
        try:
            input_text = input("Enter your input prompt: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
```
**Lines 67-70:** Prompts the user to type something. `EOFError` is raised when input is piped from a file and runs out. `KeyboardInterrupt` is raised when the user presses Ctrl+C. Both cause a graceful exit.

```python
        if console is not None:
            with console.status("[green]Generating response...[/green]", spinner="dots"):
                model_response = _run_inference()
        else:
            model_response = _run_inference()
```
**Lines 84-90:** If a Rich console object was provided, shows an animated spinner while waiting for the AI to respond. Otherwise just runs inference silently. `[green]...[/green]` is Rich's markup for green-colored text.

```python
        for g in model_response:
            print("------------")
            print(repr(g))
            print("------------")
```
**Lines 92-95:** Prints each response between separator lines. `repr(g)` prints the full Python representation of the response object (showing all its fields), useful for debugging.

---

```python
def infer(config, inputs, inference_engine, *, input_image_bytes, system_prompt) -> list[Conversation]:
```
**Line 99:** The batch inference function. Unlike interactive mode, this takes a list of inputs all at once and returns all responses at once.

```python
    conversations = None
    if inputs is not None and len(inputs) > 0:
```
**Lines 125-126:** `conversations = None` is the default (pass no input to the engine and let the engine use whatever it wants). Only builds conversation objects if input was actually provided.

```python
        system_messages = (
            [Message(role=Role.SYSTEM, content=system_prompt)] if system_prompt else []
        )
```
**Lines 127-129:** Creates a system message if a `system_prompt` was provided. A system message is a hidden instruction to the AI like "You are a helpful assistant." The ternary expression (`X if condition else Y`) returns the list with one message if `system_prompt` is set, or an empty list if not.

```python
        if input_image_bytes is None or len(input_image_bytes) == 0:
            conversations = [
                Conversation(
                    messages=(
                        system_messages + [Message(role=Role.USER, content=content)]
                    )
                )
                for content in inputs
            ]
```
**Lines 130-138:** For text-only inputs, creates one `Conversation` per input. Each conversation has the system message (if any) plus a user message. The `for content in inputs` part is a *list comprehension* — a compact way to create a list by applying the same operation to each item.

```python
        else:
            conversations = [
                Conversation(
                    messages=(
                        system_messages
                        + [
                            Message(
                                role=Role.USER,
                                content=(
                                    [ContentItem(type=Type.IMAGE_BINARY, binary=image_bytes)
                                     for image_bytes in input_image_bytes]
                                    + [ContentItem(type=Type.TEXT, content=content)]
                                ),
                            )
                        ]
                    )
                )
                for content in inputs
            ]
```
**Lines 139-161:** For vision+text inputs, each message contains a list of `ContentItem` objects: first all the images (as binary data), then the text. This is how multimodal AI receives mixed media input.

```python
    generations = inference_engine.infer(
        input=conversations,
        inference_config=config,
    )
    return generations
```
**Lines 163-167:** Hands the prepared conversations to the inference engine and returns whatever it produces.

---

## 4. Testing the AI: `evaluate.py`

**File:** `src/oumi/evaluate.py`

This is intentionally the simplest major file — its job is to delegate to the real evaluation machinery.

```python
def evaluate(config: EvaluationConfig) -> list[dict[str, Any]]:
```
**Line 22:** Takes a configuration, runs evaluation, returns results.

```python
    evaluator = Evaluator()
    results: list[EvaluationResult] = evaluator.evaluate(config)
    return [result.task_result for result in results]
```
**Lines 32-34:** Creates an `Evaluator` object (which wraps the LM Evaluation Harness library), runs it, and converts the results. `result.task_result` extracts the plain dictionary of scores from the full `EvaluationResult` object — a simpler format for the caller to work with.

The list comprehension `[result.task_result for result in results]` means "make a new list where each item is the `task_result` of each `result`."

---

## 5. Error Vocabulary: `exceptions.py`

**File:** `src/oumi/exceptions.py`

This file defines the custom error types Oumi uses. Instead of generic errors, Oumi throws specific named errors so users and developers know exactly what went wrong.

```python
"""This module is intentionally free of heavy dependencies...
so that it can be imported cheaply in lightweight entry-points such as the CLI."""
```
**Lines 17-19:** Explains the design philosophy: error types are defined separately from all the heavy AI libraries so the CLI can start instantly without loading PyTorch, etc.

---

```python
class OumiConfigError(Exception):
    """Raised for invalid or inconsistent configuration (paths, values, structure)."""
```
**Lines 27-28:** The base class for all configuration errors. `Exception` is Python's built-in error base class — `OumiConfigError` *extends* it with a more specific meaning. Think of this like a category label.

```python
class OumiConfigTypeError(OumiConfigError):
    """Raised when a loaded config is not an instance of the expected class."""
    def __init__(self, config_type: type, config_value: Any):
        self.config_type = config_type
        self.config_value = config_value
        super().__init__(
            f"Expected config of type {config_type.__name__}, "
            f"got {type(config_value).__name__}"
        )
```
**Lines 31-41:** A more specific error — raised when you load a config file but it turns out to be the wrong *type* of config (e.g., you tried to use an inference config for training). Stores both what was expected and what was actually received. `super().__init__(...)` calls the parent class's constructor with a formatted error message.

```python
class OumiConfigParsingError(OumiConfigError):
    def __init__(self, cause: "OmegaConfBaseException"):
        key = getattr(cause, "full_key", None) or getattr(cause, "key", None)
        self.config_key: str | None = str(key) if key is not None else None
        msg = getattr(cause, "msg", None) or str(cause)
        if self.config_key:
            super().__init__(f"Config error at '{self.config_key}': {msg}")
        else:
            super().__init__(f"Config error: {msg}")
```
**Lines 44-60:** Wraps OmegaConf parsing errors (raw YAML parsing failures) into friendlier messages. `getattr(cause, "full_key", None)` tries to extract which YAML key caused the error — if found, the error message will say exactly which line in your config file is wrong.

```python
class HardwareException(Exception):
    """An exception thrown for invalid hardware configurations."""
```
**Lines 63-64:** A completely separate error type for hardware problems (e.g., asking for 8 GPUs when only 4 are available).

---

## 6. The Recipe Card: `training_config.py`

**File:** `src/oumi/core/configs/training_config.py`

This file defines `TrainingConfig` — the master configuration class. Every training run uses one of these.

```python
@dataclass
class TrainingConfig(BaseConfig):
```
**Lines 36-37:** `@dataclass` is a Python decorator that automatically generates standard methods for the class (equality comparison, string representation, etc.). `BaseConfig` is Oumi's base class that adds YAML loading/saving capabilities.

```python
    data: DataParams = field(default_factory=DataParams)
    """Parameters for the dataset."""
```
**Lines 38-39:** The `data` field holds all dataset-related settings (which files to load, how to format them, etc.). `field(default_factory=DataParams)` means "if not specified, create a default empty `DataParams`." The docstring explains what this field is for.

```python
    model: ModelParams = field(default_factory=ModelParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    peft: PeftParams = field(default_factory=PeftParams)
    fsdp: FSDPParams = field(default_factory=FSDPParams)
    deepspeed: DeepSpeedParams = field(default_factory=DeepSpeedParams)
```
**Lines 48-88:** Five more sub-configs, each handling a different concern:
- `model` — which AI model to use, its architecture
- `training` — learning rate, batch size, epochs, optimizer
- `peft` — LoRA rank, adapter settings
- `fsdp` — how to shard the model across GPUs
- `deepspeed` — memory offloading and ZeRO optimization settings

```python
    def __post_init__(self):
        """Verifies/populates params."""
        if self.model.compile:
            raise OumiConfigError(
                "Use `training.compile` instead of `model.compile`..."
            )
```
**Lines 90-95:** `__post_init__` runs automatically right after the object is created. It validates the configuration, catching common mistakes. This particular check prevents a confusing misconfiguration where someone puts the `compile` setting in the wrong section.

---

## 7. Trainer Choices: `training_params.py`

**File:** `src/oumi/core/configs/params/training_params.py`

This file defines the `TrainerType` enum — the list of all supported training algorithms.

```python
class TrainerType(Enum):
    """Enum representing the supported trainers."""
```
**Lines 38-39:** An `Enum` is a fixed collection of named constants. Once defined, you can only choose from these options.

```python
    TRL_SFT = "trl_sft"
    """Supervised fine-tuning trainer from `trl` library."""
```
**Lines 41-43:** SFT (Supervised Fine-Tuning) — the most common training method. The AI learns from pairs of (instruction, ideal response). The string `"trl_sft"` is what you write in your YAML config file.

```python
    TRL_DPO = "trl_dpo"
    """Direct Preference Optimization trainer."""
```
**Lines 48-51:** DPO — teaches the AI to *prefer* good responses over bad ones by showing it pairs of (winner, loser) responses.

```python
    TRL_KTO = "trl_kto"
    """Kahneman-Tversky Optimization trainer."""
```
**Lines 55-60:** KTO — a variant that only needs a single response labeled as "good" or "bad" (not pairs), making data collection easier.

```python
    TRL_GRPO = "trl_grpo"
    """Group Relative Policy Optimization trainer."""
```
**Lines 62-69:** GRPO — trains the AI to reason step-by-step (like solving math). The AI generates multiple answers, a reward function scores them, and the AI learns to produce higher-scoring answers.

```python
    TRL_GKD = "trl_gkd"
    """Generalized Knowledge Distillation trainer."""
```
**Lines 71-82:** GKD — a "teacher-student" approach where a large AI (teacher) corrects the outputs of a smaller AI (student) in real time.

```python
    OUMI = "oumi"
    """Custom generic trainer implementation."""
```
**Lines 102-107:** Oumi's own built-in trainer with the most flexibility for custom experimentation.

```python
    VERL_GRPO = "verl_grpo"
    """Group Relative Policy Optimization trainer from `verl` library."""
```
**Lines 109-116:** The most advanced option — GRPO using the VERL framework, which runs on Ray for massive-scale distributed reinforcement learning.

---

## 8. The Message Format: `conversation.py`

**File:** `src/oumi/core/types/conversation.py`

This file defines the data structures that represent a conversation between a user and an AI. Everything flowing through Oumi's inference system is ultimately built from these types.

```python
class Role(str, Enum):
    """Role of the entity sending the message."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
```
**Lines 27-40:** Every message has a `Role` — who sent it. `SYSTEM` is the hidden instruction. `USER` is the human. `ASSISTANT` is the AI. `TOOL` is the result from an external tool call (like a calculator or web search).

```python
class Type(str, Enum):
    TEXT = "text"
    IMAGE_PATH = "image_path"
    IMAGE_URL = "image_url"
    IMAGE_BINARY = "image_binary"
```
**Lines 51-64:** Every piece of content has a `Type`. A message can contain text, or an image from a file path, or an image from a web URL, or raw image bytes.

```python
class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"
```
**Lines 75-94:** When the AI stops generating text, there's always a reason. `STOP` = natural ending. `LENGTH` = hit the maximum allowed response length. `TOOL_CALLS` = the AI wants to call an external tool. `CONTENT_FILTER` = the safety system blocked the output. This field tells downstream code exactly why the AI stopped.

These three simple enums — `Role`, `Type`, and `FinishReason` — form the vocabulary that every single conversation in Oumi uses, from interactive chat to batch inference to evaluation.

---

## 9. The Factory Hub: `builders/__init__.py`

**File:** `src/oumi/builders/__init__.py`

This file is a catalog of all factory functions — functions that *create* the components of a training run.

```python
"""Builders module for the Oumi library.

This module provides builder functions to construct and configure
different components of the Oumi framework, including datasets, models,
optimizers, and trainers.

The builder functions encapsulate the complexity of creating these components,
allowing for easier setup and configuration of machine learning experiments.
"""
```
**Lines 15-23:** The module docstring explains the design philosophy: instead of each part of the code knowing how to create every kind of model/dataset/trainer, that knowledge is centralized into "builder" functions.

```python
from oumi.builders.callbacks import build_training_callbacks
from oumi.builders.collators import build_collator_from_config, build_data_collator, ...
from oumi.builders.data import build_dataset, build_dataset_mixture
from oumi.builders.models import build_chat_template, build_model, build_peft_model, build_tokenizer, ...
from oumi.builders.training import build_trainer
```
**Lines 25-50:** Each line imports specific factory functions from the corresponding file. `build_model` creates the AI model. `build_tokenizer` creates the text-to-numbers converter. `build_trainer` creates the training loop. `build_dataset_mixture` combines multiple datasets. `build_training_callbacks` creates the event hooks for checkpointing/logging.

```python
__all__ = [
    "build_chat_template",
    "build_dataset_mixture",
    ...
]
```
**Lines 52-73:** The official public API of the builders module — the exact list of names that are exported.

---

## 10. Building Models: `builders/models.py`

**File:** `src/oumi/builders/models.py`

This file contains the logic for loading an AI model into memory.

```python
try:
    import liger_kernel.transformers
except ImportError:
    liger_kernel = None
```
**Lines 42-45:** A "soft import" — tries to import `liger_kernel` (a GPU optimization library that makes training faster), but doesn't crash if it's not installed. If missing, `liger_kernel` is set to `None` and the code later checks `if liger_kernel is not None` before using it.

```python
try:
    import onebitllms
    from onebitllms import replace_linear_with_bitnet_linear
except ImportError:
    onebitllms = None
```
**Lines 47-52:** Same pattern for `onebitllms` — a library for "BitNet" models that use 1-bit weights (extremely compressed models). Optional — only used if installed.

```python
def build_model(
    model_params: ModelParams,
    peft_params: PeftParams | None = None,
    **kwargs,
) -> nn.Module:
```
**Lines 55-59:** The main model builder. Returns `nn.Module` — PyTorch's universal base class for all neural networks. No matter what model you specify in the config, what comes back is an `nn.Module`.

```python
    if is_custom_model(model_params.model_name):
        model = build_oumi_model(...)
    else:
        model = build_huggingface_model(...)
```
**Lines 70-80:** If the model name refers to a custom Oumi model (like a tiny test model), use Oumi's own loader. Otherwise, use HuggingFace's loader, which handles the vast majority of models (Llama, Qwen, Phi, Gemma, etc.).

---

## 11. Loading Datasets: `builders/data.py`

**File:** `src/oumi/builders/data.py`

This file builds the training and validation datasets from the configuration.

```python
DatasetType = TypeVar("DatasetType", datasets.Dataset, datasets.IterableDataset)
```
**Line 37:** A "TypeVar" is a placeholder type. `DatasetType` can be either a regular HuggingFace `Dataset` (all data in memory) or an `IterableDataset` (streamed one batch at a time — useful for enormous datasets that don't fit in RAM).

```python
def build_dataset_mixture(
    data_params: DataParams,
    tokenizer: BaseTokenizer | None,
    dataset_split: DatasetSplit,
    seq_length: int | None = None,
    seed: int | None = None,
) -> DatasetType | PretrainingAsyncTextDataset:
```
**Lines 40-46:** The main dataset builder. `seq_length` controls how long each text example is (shorter = faster training, longer = more context for the AI). `seed` sets a random seed so datasets are mixed in the same order if you run training twice (reproducibility).

```python
    dataset_split_params: DatasetSplitParams = data_params.get_split(dataset_split)
    if dataset_split_params.use_torchdata:
        ...
        logger.warning("Using torchdata preprocessing pipeline. This is currently in beta...")
        return build_oumi_dataset(...)
```
**Lines 62-74:** An experimental feature flag — if `use_torchdata=True` in the config, uses an alternative data pipeline (still in beta testing). Prints a warning to make sure users know it's not production-ready.

```python
    is_packed = _is_mixture_packed(dataset_split_params)
    datasets = [
```
**Lines 78-80:** "Packing" means fitting multiple short examples end-to-end into a single fixed-length sequence (to avoid wasting space on padding). This line checks if packing is requested.

---

## Putting It All Together

Here is the complete flow from typing `oumi train -c config.yaml` to a trained model:

```
1. CLI reads config.yaml → creates a TrainingConfig object
         (training_config.py defines the structure)

2. train() function is called in train.py
         ↓
3. Tokenizer is built             (builders/models.py → build_tokenizer)
4. Datasets are loaded            (builders/data.py → build_dataset_mixture)
5. Model is loaded into GPU RAM   (builders/models.py → build_model)
6. PEFT adapters are attached     (builders/models.py → build_peft_model)
7. Trainer object is created      (builders/training.py → build_trainer)
         ↓
8. barrier() — all GPUs synchronize
9. trainer.train() — the AI studies the data
         ↓
10. Checkpoints saved periodically during training
11. Final model saved at the end
12. GPU communication network shut down
```

Every step in this chain corresponds to specific lines explained above. The config file is the input, the trained model weights are the output, and `train.py` is the conductor that makes it all happen in the right order.

---

*Document generated May 2026.*
