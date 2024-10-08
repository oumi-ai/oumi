"""Minimal multi-modal training script with CLI arguments and custom collator.

Run the script using:
   python scripts/benchmarks/minimal_multimodal_training.py \
    --model-name<model_name> --dataset-name <dataset_name>

For multi-GPU training, use torchrun:
   torchrun --standalone --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) \
        scripts/benchmarks/minimal_multimodal_training.py \
            --model-name <model_name> --dataset-name <dataset_name>

Working configs:
    --model-name Salesforce/blip2-opt-2.7b --dataset-name coco_captions
    --model-name Salesforce/blip2-opt-2.7b --dataset-name nlphuji/flickr30k
    --model-name llava-hf/llava-1.5-7b-hf --dataset-name coco_captions --test-fsdp
    --model-name llava-hf/llava-1.5-7b-hf --dataset-name nlphuji/flickr30k --test-fsdp
"""

from enum import Enum
from pprint import pformat
from typing import Dict, List, Optional

import torch
import typer
from transformers import AutoProcessor

from oumi.builders import (
    build_chat_template,
    build_data_collator,
    build_dataset,
    build_model,
)
from oumi.core.configs import (
    FSDPParams,
    ModelParams,
    TrainingParams,
)
from oumi.core.distributed import (
    cleanup_distributed,
    init_distributed,
    is_distributed,
    is_local_process_zero,
)
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.trainers.oumi_trainer import Trainer
from oumi.utils.str_utils import sanitize_run_name
from oumi.utils.torch_utils import (
    log_model_summary,
)


class ModelName(str, Enum):
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    LLAVA = "llava-hf/llava-1.5-7b-hf"
    QWEN = "Qwen/Qwen2-VL-2B-Instruct"  # not supported by transformers==4.43.4
    CHAMELEON = "facebook/chameleon-7b"
    PALIGEMMA = "google/paligemma-3b-mix-224"


_FREEZE_LAYERS_MAP: Dict[ModelName, List[str]] = {
    ModelName.BLIP2: ["vision_model"],
    ModelName.LLAVA: ["vision_tower"],
    ModelName.QWEN: ["visual"],
    ModelName.CHAMELEON: ["model.vqmodel"],  # FIXME Freeze nested layers OPE-505
    ModelName.PALIGEMMA: ["vision_tower"],
}


def _get_freeze_layers(model_name: ModelName) -> List[str]:
    result = []
    if model_name in _FREEZE_LAYERS_MAP:
        result = _FREEZE_LAYERS_MAP[model_name]
        print(f"Frozen layers: {result}")
    else:
        print(f"No frozen layers defined for {model_name}!")
    return result


class DatasetName(str, Enum):
    COCO = "coco_captions"
    FLICKR = "nlphuji/flickr30k"
    LLAVA_INSTRUCT_MIX_VSFT = "HuggingFaceH4/llava-instruct-mix-vsft"


def _get_default_dataset_split(dataset_name: DatasetName) -> str:
    if dataset_name == DatasetName.FLICKR:
        # The dataset only has "test" split.
        return "test"
    return "train"


def test_multimodal_trainer(
    model_name: ModelName = ModelName.BLIP2,
    dataset_name: DatasetName = DatasetName.COCO,
    batch_size: int = 2,
    max_steps: int = 20,
    logging_steps: int = 5,
    split: Optional[str] = None,
    test_inference: bool = False,
    test_save_state: bool = False,
    test_fsdp: bool = False,
):
    """Minimal multi-modal training loop."""
    if is_distributed():
        print("Initializing distributed process group")
        init_distributed()
    else:
        print("Not initializing distributed process group")

    if not split:
        split = _get_default_dataset_split(dataset_name)

    #
    # Init model, processor, and dataset
    #
    model_params = ModelParams(
        model_name=model_name.value,
        torch_dtype_str="float16",
        trust_remote_code=True,
        freeze_layers=_get_freeze_layers(model_name),  # TODO: fix freeze + fsdp
    )
    if is_local_process_zero():
        print(f"ModelParams:\n{pformat(model_params)}")

    model = build_model(model_params)
    processor = AutoProcessor.from_pretrained(model_name.value)
    tokenizer: BaseTokenizer = processor.tokenizer

    # TODO: assign the right chat template for each model
    # For now, we use the LLaVA chat template for all models
    # NOTE: We can't use the original model's template because
    # oumi will feed it an array of `oumi.core.types.turn.Message`
    # objects (vs model-specific Python dict).
    chat_template = build_chat_template("llava")
    processor.chat_template = chat_template
    tokenizer.chat_template = chat_template

    dataset = build_dataset(
        dataset_name=str(dataset_name.value),
        tokenizer=processor.tokenizer,
        split=split,
        dataset_kwargs=dict(processor=processor, limit=100),
        trust_remote_code=True,
        experimental_use_torch_datapipes=True,
    )

    #
    # Set up training parameters
    #
    fsdp_params = FSDPParams(
        enable_fsdp=is_distributed() and test_fsdp, cpu_offload=True
    )

    run_name = sanitize_run_name(
        f"multimodal_test_{model_name.value.split('/')[-1]}_{dataset_name.value}"
    )

    training_params = TrainingParams(
        output_dir=f"output/{run_name}",
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        save_steps=0,
        optimizer="sgd",
        learning_rate=2e-5,
        gradient_accumulation_steps=1,
        log_model_summary=False,
        logging_steps=logging_steps,
        include_performance_metrics=True,
    )

    if is_local_process_zero():
        print(f"TrainingParams:\n{pformat(training_params)}")
        if training_params.log_model_summary:
            log_model_summary(model)

    # Initialize trainer with custom collator
    collator = build_data_collator(
        collator_name="vision_language",
        tokenizer=tokenizer,
        max_length=model_params.model_max_length,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_params,
        train_dataset=dataset,
        fsdp_params=fsdp_params,
        data_collator=collator,
    )

    #
    # Train
    #
    trainer.train()
    if test_save_state:
        trainer.save_state()

    #
    # Test inference
    #
    if test_inference:
        test_input = dataset[0]
        with torch.no_grad():
            output = trainer.model(**test_input)

        print("Test output:", output.keys())
        print("Test output shapes:", {k: v.shape for k, v in output.items()})

    if is_distributed():
        cleanup_distributed()

    print(
        f"Multi-modal training test successful with model {model_name} and "
        f"dataset {dataset_name}!"
    )

    return True


if __name__ == "__main__":
    typer.run(test_multimodal_trainer)
