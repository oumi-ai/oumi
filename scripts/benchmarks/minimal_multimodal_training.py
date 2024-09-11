"""Minimal multi-modal training script with CLI arguments and custom collator.

Run the script using:
   python scripts/benchmarks/minimal_multimodal_training.py \
    --model_name <model_name> --dataset <dataset_name>

For multi-GPU training, use torchrun:
   torchrun --standalone --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) \
        scripts/benchmarks/minimal_multimodal_training.py \
            --model_name <model_name> --dataset <dataset_name>
"""

import argparse

import numpy as np
import torch
from transformers import AutoProcessor, DataCollatorWithPadding

from lema.builders.models import build_model
from lema.core.configs.params.model_params import ModelParams
from lema.core.configs.params.training_params import TrainingParams
from lema.core.datasets import VisionLanguageSftDataset
from lema.core.distributed import cleanup_distributed, init_distributed, is_distributed
from lema.core.trainers.lema_trainer import Trainer
from lema.datasets import (
    COCOCaptionsDataset,
    Flickr30kDataset,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Minimal multi-modal training script")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip2-opt-2.7b",
        choices=[
            "Salesforce/blip2-opt-2.7b",
            "llava-hf/llava-1.5-7b-hf",
            "Qwen/Qwen2-VL-2B-Instruct",
            "facebook/chameleon-7b",
            "google/paligemma-3b-mix-224",
        ],
        help="Name of the multi-modal model to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="coco_captions",
        choices=["coco_captions", "nlphuji/flickr30k"],
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Per-device training batch size"
    )
    parser.add_argument(
        "--max_steps", type=int, default=10, help="Maximum number of training steps"
    )
    return parser.parse_args()


def get_dataset(
    dataset_name: str, processor, limit: int = 100
) -> VisionLanguageSftDataset:
    """Get a dataset for multi-modal training."""
    if dataset_name == "coco_captions":
        return COCOCaptionsDataset(split="train", processor=processor, limit=limit)
    elif dataset_name == "nlphuji/flickr30k":
        return Flickr30kDataset(split="train", processor=processor, limit=limit)
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


def test_multimodal_trainer(args):
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
        model_name=args.model_name,
        torch_dtype_str="float16",
        trust_remote_code=True,
        # freeze_layers=["vision_model"],
    )
    model = build_model(model_params)
    processor = AutoProcessor.from_pretrained(args.model_name)

    # TODO: Add chat template to processor
    # processor.chat_template = LLAVA_CHAT_TEMPLATE
    # processor.tokenizer.chat_template = LLAVA_CHAT_TEMPLATE

    # TODO: OPE-357 Add builder for collator
    collator = MultiModalCollator(processor)
    dataset = get_dataset(args.dataset, processor)

    #
    # Set up training parameters
    #
    # fsdp_params = FSDPParams(enable_fsdp=False, cpu_offload=True)

    training_params = TrainingParams(
        output_dir=f"output/multimodal_test_{args.model_name.split('/')[-1]}_{args.dataset}",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=args.max_steps,
        save_steps=0,
        learning_rate=args.learning_rate,
        log_model_summary=True,
        logging_steps=1,
        include_performance_metrics=True,
    )

    # Initialize trainer with custom collator
    # collator = MultiModalCollator(processor)
    trainer = Trainer(
        model=model,
        tokenizer=processor.tokenizer,
        args=training_params,
        train_dataset=dataset,
        # fsdp_params=fsdp_params,
        data_collator=collator,
    )

    #
    # Train
    #
    trainer.train()
    # trainer.save_state()

    #
    # Test inference
    #
    test_input = dataset[0]
    with torch.no_grad():
        output = trainer.model(**test_input)

    print("Test output:", output.keys())
    print("Test output shapes:", {k: v.shape for k, v in output.items()})

    if is_distributed():
        cleanup_distributed()

    return True


if __name__ == "__main__":
    args = parse_args()
    test_multimodal_trainer(args)
    print(
        f"Multi-modal training test successful with model {args.model_name} and "
        "dataset {args.dataset}!"
    )
