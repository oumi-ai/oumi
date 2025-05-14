"""Minimal multi-modal training script with CLI arguments and custom collator.

Run the script using:
   python scripts/benchmarks/minimal_multimodal_training.py \
    --model-name<model_name> --dataset-name <dataset_name>

For multi-GPU training, use torchrun:
   torchrun --standalone --nproc-per-node=$(nvidia-smi --list-gpus | wc -l) \
        scripts/benchmarks/minimal_multimodal_training.py \
            --model-name <model_name> --dataset-name <dataset_name>

Working configs:
    --model-name Salesforce/blip2-opt-2.7b --dataset-name merve/vqav2-small
    --model-name Salesforce/blip2-opt-2.7b --dataset-name nlphuji/flickr30k
    --model-name Qwen/Qwen2-VL-2B-Instruct --dataset-name merve/vqav2-small
    --model-name Qwen/Qwen2-VL-2B-Instruct --dataset-name nlphuji/flickr30k
    --model-name Qwen/Qwen2.5-VL-3B-Instruct --dataset-name merve/vqav2-small
    --model-name llava-hf/llava-1.5-7b-hf --dataset-name merve/vqav2-small --test-fsdp
    --model-name llava-hf/llava-1.5-7b-hf --dataset-name nlphuji/flickr30k --test-fsdp


"""

from enum import Enum
from pprint import pformat
from typing import Final, NamedTuple, Optional

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
    LLAVA = "llava-hf/llava-1.5-7b-hf"
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    LLAMA_11B_VISION_INSTRUCT = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    QWEN2_VL = "Qwen/Qwen2-VL-2B-Instruct"
    QWEN2_5_3B_VL = "Qwen/Qwen2.5-VL-3B-Instruct"
    CHAMELEON = "facebook/chameleon-7b"
    PALIGEMMA = "google/paligemma-3b-mix-224"
    PHI3_VISION = "microsoft/Phi-3-vision-128k-instruct"  # requires flash-attn
    MOLMOE_1B = "allenai/MolmoE-1B-0924"
    SMOLVLM = "HuggingFaceTB/SmolVLM-Instruct"


_DEFAULT_MLLM_CHAT_TEMPLATE: Final[str] = "llava"
_DEFAULT_MLLM_COLLATOR: Final[str] = "vision_language_with_padding"


class ModelInfo(NamedTuple):
    chat_template: str
    freeze_layers: list[str]
    supports_multiple_images: bool = False
    collator_name: str = _DEFAULT_MLLM_COLLATOR


_MODELS_MAP: dict[ModelName, ModelInfo] = {
    ModelName.BLIP2: ModelInfo(
        chat_template="default",
        freeze_layers=["vision_model"],
    ),
    ModelName.LLAVA: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["vision_tower"],
    ),
    ModelName.QWEN2_VL: ModelInfo(
        chat_template="qwen2-vl-instruct",
        freeze_layers=["visual"],
        collator_name="vision_language_sft",
    ),
    ModelName.QWEN2_5_3B_VL: ModelInfo(
        chat_template="qwen2-vl-instruct",
        freeze_layers=["visual"],
    ),
    ModelName.CHAMELEON: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["model.vqmodel"],
    ),
    ModelName.PALIGEMMA: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["vision_tower"],
    ),
    ModelName.PHI3_VISION: ModelInfo(
        chat_template="phi3-instruct",
        freeze_layers=["model.vision_embed_tokens"],
    ),
    ModelName.LLAMA_11B_VISION_INSTRUCT: ModelInfo(
        chat_template="llama3-instruct",
        freeze_layers=["vision_model"],
        supports_multiple_images=True,
    ),
    ModelName.MOLMOE_1B: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["model.vision_backbone"],
    ),
    ModelName.SMOLVLM: ModelInfo(
        chat_template=_DEFAULT_MLLM_CHAT_TEMPLATE,
        freeze_layers=["vision_model"],
    ),
}


def _get_freeze_layers(model_name: ModelName) -> list[str]:
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
        result = _DEFAULT_MLLM_CHAT_TEMPLATE
        print(f"No chat template defined for {model_name}! Defaulting to '{result}'...")
    return result or _DEFAULT_MLLM_CHAT_TEMPLATE


def _get_collator_name(model_name: ModelName) -> str:
    result = ""
    if model_name in _MODELS_MAP:
        result = _MODELS_MAP[model_name].collator_name
        print(f"Collator: {result}")
    else:
        result = _DEFAULT_MLLM_COLLATOR
        print(f"No collator defined for {model_name}! Defaulting to '{result}'...")
    return result or _DEFAULT_MLLM_COLLATOR


def _supports_multiple_images(model_name: ModelName) -> bool:
    result = False
    if model_name in _MODELS_MAP:
        result = _MODELS_MAP[model_name].supports_multiple_images
    return result


class DatasetName(str, Enum):
    MERVE_VQAV2_SMALL = "merve/vqav2-small"
    LLAVA_INSTRUCT_MIX_VSFT = "HuggingFaceH4/llava-instruct-mix-vsft"
    FLICKR = "nlphuji/flickr30k"
    COCO = "coco_captions"
    MNIST_SFT = "mnist_sft"
    DOCMATIX = "HuggingFaceM4/Docmatix"


class _DatasetInfo(NamedTuple):
    default_split: str
    default_subset: Optional[str]
    contains_multiple_images: bool


def _get_dataset_info(dataset_name: DatasetName) -> _DatasetInfo:
    default_split = "train"
    if dataset_name in (DatasetName.FLICKR,):
        # The dataset only has "test" split.
        default_split = "test"
    elif dataset_name in (DatasetName.MERVE_VQAV2_SMALL,):
        default_split = "validation"

    contains_multiple_images: bool = False

    default_subset: Optional[str] = None
    if dataset_name in (DatasetName.DOCMATIX,):
        # The only non-giant subset in the dataset
        default_subset = "zero-shot-exp"
        contains_multiple_images = True

    return _DatasetInfo(
        default_split=default_split,
        default_subset=default_subset,
        contains_multiple_images=contains_multiple_images,
    )


def test_multimodal_trainer(
    model_name: ModelName = ModelName.BLIP2,
    dataset_name: DatasetName = DatasetName.COCO,
    batch_size: int = 2,
    max_steps: int = 20,
    optimizer: str = "sgd",
    logging_steps: int = 5,
    split: Optional[str] = None,
    subset: Optional[str] = None,
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

    dataset_info = _get_dataset_info(dataset_name)

    if not split:
        split = dataset_info.default_split
    if not subset:
        subset = dataset_info.default_subset

    if dataset_info.contains_multiple_images and not _supports_multiple_images(
        model_name
    ):
        print(
            f"Using multi-image dataset {dataset_name} "
            f"with model {model_name} that doesn't support multiple images! "
            "Training may FAIL!"
        )

    #
    # Init model, processor, and dataset
    #
    model_params = ModelParams(
        model_name=model_name.value,
        torch_dtype_str="bfloat16",
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

    collator_name = _get_collator_name(model_name)
    dataset_kwargs = dict(processor=processor, limit=100)
    if collator_name == "vision_language_sft":
        dataset_kwargs["return_conversations"] = True
    dataset = build_dataset(
        dataset_name=str(dataset_name.value),
        tokenizer=tokenizer,
        subset=subset,
        split=split,
        dataset_kwargs=dataset_kwargs,
        trust_remote_code=True,
        use_torchdata=True,
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
        optimizer=(optimizer or "sgd"),
        learning_rate=2e-5,
        warmup_steps=int(max(10, 0.2 * max_steps)),
        max_grad_norm=10,
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

    collator_kwargs = {}
    if collator_name == "vision_language_sft":
        collator_kwargs["processor_name"] = model_name.value
        collator_kwargs["processor_kwargs"] = model_params.processor_kwargs
        collator_kwargs["trust_remote_code"] = model_params.trust_remote_code

    # Initialize trainer with custom collator
    collator = build_data_collator(
        collator_name=collator_name,
        tokenizer=tokenizer,
        max_length=model_params.model_max_length,
        label_ignore_index=constants.LABEL_IGNORE_INDEX,
        **collator_kwargs,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
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
