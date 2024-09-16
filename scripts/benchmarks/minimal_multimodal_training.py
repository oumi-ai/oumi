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
    --model-name Salesforce/blip2-opt-2.7b --dataset-name flickr30k
    --model-name llava-hf/llava-1.5-7b-hf --dataset-name coco_captions --test_fsdp
    --model-name llava-hf/llava-1.5-7b-hf --dataset-name flickr30k --test_fsdp
"""

from enum import Enum

import numpy as np
import torch
import typer
from transformers import AutoProcessor, DataCollatorWithPadding

from oumi.builders.models import build_chat_template, build_model
from oumi.core.configs import FSDPParams, ModelParams, TrainingParams
from oumi.core.distributed import cleanup_distributed, init_distributed, is_distributed
from oumi.core.trainers.lema_trainer import Trainer
from oumi.datasets import COCOCaptionsDataset, Flickr30kDataset
from oumi.utils.str_utils import sanitize_run_name


class ModelName(str, Enum):
    BLIP2 = "Salesforce/blip2-opt-2.7b"
    LLAVA = "llava-hf/llava-1.5-7b-hf"
    QWEN = "Qwen/Qwen2-VL-2B-Instruct"
    CHAMELEON = "facebook/chameleon-7b"
    PALIGEMMA = "google/paligemma-3b-mix-224"


class DatasetName(str, Enum):
    COCO = "coco_captions"
    FLICKR = "nlphuji/flickr30k"


def get_dataset(dataset_name: DatasetName, processor, limit: int = 100):
    """Get a dataset for multi-modal training."""
    if dataset_name == DatasetName.COCO:
        return COCOCaptionsDataset(
            split="train", processor=processor, limit=limit, trust_remote_code=True
        )
    elif dataset_name == DatasetName.FLICKR:
        return Flickr30kDataset(
            split="test", processor=processor, limit=limit, trust_remote_code=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


class MultiModalCollator:
    def __init__(self, processor):
        """Custom collator for multi-modal training."""
        self.processor = processor
        self.default_collator = DataCollatorWithPadding(
            tokenizer=self.processor.tokenizer,
            padding=True,
            max_length=1024,
        )

    def __call__(self, batch):
        """Custom collator for multi-modal training.

        Args:
            batch: List of batch items.

        Returns:
            Dict[str, torch.Tensor]: Processed batch.
        """
        images = [item["pixel_values"] for item in batch]
        text_inputs = [item["input_ids"] for item in batch]

        # collate batch images
        pixel_values = self.collate_images(images)

        # collate batch prompts
        text_inputs = self.default_collator({"input_ids": text_inputs})  # type: ignore

        # Combine all inputs
        combined_batch = {
            "pixel_values": pixel_values,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
        }

        # Add labels if present
        if "labels" in batch[0]:
            combined_batch["labels"] = text_inputs["input_ids"]

        return combined_batch

    def collate_images(self, images) -> torch.Tensor:
        """Collate images for multi-modal training.

        Args:
            images: List of images to collate.

        Returns:
            torch.Tensor: Batch of processed images.
        """
        if isinstance(images[0], torch.Tensor):
            return torch.stack(images)
        elif isinstance(images[0], np.ndarray):
            return torch.stack([torch.from_numpy(img) for img in images])
        elif isinstance(images[0], list):
            return torch.tensor(images)
        else:
            raise ValueError(f"Unsupported image type: {type(images[0])}")


def test_multimodal_trainer(
    model_name: ModelName = ModelName.BLIP2,
    dataset_name: DatasetName = DatasetName.COCO,
    batch_size: int = 2,
    max_steps: int = 10,
    logging_steps: int = 1,
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

    #
    # Init model, processor, and dataset
    #
    model_params = ModelParams(
        model_name=model_name.value,
        torch_dtype_str="float16",
        trust_remote_code=True,
        # freeze_layers=["vision_model"],  # TODO: fix freeze + fsdp
    )
    model = build_model(model_params)
    processor = AutoProcessor.from_pretrained(model_name.value)

    # TODO: assign the right chat template for each model
    # For now, we use the LLaVA chat template for all models
    chat_template = build_chat_template("llava")
    processor.chat_template = chat_template
    processor.tokenizer.chat_template = chat_template

    collator = MultiModalCollator(processor)
    dataset = get_dataset(dataset_name, processor)

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
        gradient_accumulation_steps=1,
        log_model_summary=False,
        logging_steps=logging_steps,
        include_performance_metrics=True,
    )

    # Initialize trainer with custom collator
    collator = MultiModalCollator(processor)
    trainer = Trainer(
        model=model,
        tokenizer=processor.tokenizer,
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
