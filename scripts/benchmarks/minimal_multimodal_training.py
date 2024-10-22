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
from typing import Dict, List, NamedTuple, Optional

import torch
import typer

import oumi.core.constants as constants
from oumi.builders import (
    build_data_collator,
    build_dataset,
    build_model,
    build_processor,
    build_tokenizer,
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
from oumi.core.processors.base_processor import BaseProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.trainers.oumi_trainer import Trainer
from oumi.utils.str_utils import sanitize_run_name
from oumi.utils.torch_utils import (
    log_model_summary,
)


class ModelName(str, Enum):
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    LLAVA = "llava-hf/llava-1.5-7b-hf"
    QWEN = "Qwen/Qwen2-VL-2B-Instruct"
    CHAMELEON = "facebook/chameleon-7b"
    PALIGEMMA = "google/paligemma-3b-mix-224"
    PHI3_VISION = "microsoft/Phi-3-vision-128k-instruct"  # requires flash-attn
    LLAMA_11B_VISION_INSTRUCT = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    MOLMOE_1B = "allenai/MolmoE-1B-0924"


class ModelInfo(NamedTuple):
    chat_template: str
    freeze_layers: List[str]


_DEFAULT_MLLM_CHAT_TEMPLATE = "llava"

_MODELS_MAP: Dict[ModelName, ModelInfo] = {
    ModelName.BLIP2: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE, freeze_layers=["vision_model"]
    ),
    ModelName.LLAVA: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE, freeze_layers=["vision_tower"]
    ),
    ModelName.QWEN: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE, freeze_layers=["visual"]
    ),
    ModelName.CHAMELEON: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["model.vqmodel"],  # FIXME Freeze nested layers OPE-505
    ),
    ModelName.PALIGEMMA: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE, freeze_layers=["vision_tower"]
    ),
    ModelName.PHI3_VISION: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=[
            "model.vision_embed_tokens"
        ],  # FIXME Freeze nested layers OPE-505
    ),
    ModelName.LLAMA_11B_VISION_INSTRUCT: ModelInfo(
        chat_template="llama3-instruct", freeze_layers=["vision_model"]
    ),
    ModelName.MOLMOE_1B: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["model.vision_backbone"],  # FIXME Freeze nested layers OPE-505
    ),
}


def _get_freeze_layers(model_name: ModelName) -> List[str]:
    result = []
    if model_name in _MODELS_MAP:
        result = _MODELS_MAP[model_name].freeze_layers
        print(f"Frozen layers: {result}")
    else:
        print(f"No frozen layers defined for {model_name}!")
    return result


def _get_chat_template(model_name: ModelName) -> str:
    result = ""
    if model_name in _MODELS_MAP:
        result = _MODELS_MAP[model_name].chat_template
        print(f"Chat template: {result}")
    else:
        print(f"No chat templates  defined for {model_name}!")
    return result or _DEFAULT_MLLM_CHAT_TEMPLATE


class DatasetName(str, Enum):
    COCO = "coco_captions"
    FLICKR = "nlphuji/flickr30k"
    LLAVA_INSTRUCT_MIX_VSFT = "HuggingFaceH4/llava-instruct-mix-vsft"
    MERVE_VQAV2_SMALL = "merve/vqav2-small"


def _get_default_dataset_split(dataset_name: DatasetName) -> str:
    if dataset_name == DatasetName.FLICKR:
        # The dataset only has "test" split.
        return "test"
    elif dataset_name == DatasetName.MERVE_VQAV2_SMALL:
        return "validation"
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
        chat_template=_get_chat_template(model_name),
        freeze_layers=_get_freeze_layers(model_name),  # TODO: fix freeze + fsdp
    )
    if is_local_process_zero():
        print(f"ModelParams:\n{pformat(model_params)}")

    model = build_model(model_params)
    tokenizer: BaseTokenizer = build_tokenizer(model_params)
    processor: BaseProcessor = build_processor(
        model_name.value, tokenizer, trust_remote_code=True
    )

    dataset = build_dataset(
        dataset_name=str(dataset_name.value),
        tokenizer=tokenizer,
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
        optimizer="adamw_torch_fused",
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
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
        collator_name="vision_language_with_padding",
        tokenizer=tokenizer,
        max_length=model_params.model_max_length,
        label_ignore_index=constants.LABEL_IGNORE_INDEX,
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
