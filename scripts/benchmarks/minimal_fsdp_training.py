"""Minimal FSDP training loop.

This script demonstrates a minimal FSDP training loop using the Oumi framework.

Run the script using torchrun for multi-GPU training:
   torchrun --standalone --nproc-per-node=NUM_GPUS \
        scripts/benchmarks/minimal_fsdp_training.py

   Replace NUM_GPUS with the number of GPUs you want to use.

For single-GPU or CPU training, you can run the script directly:
   python scripts/benchmarks/minimal_fsdp_training.py
"""

import torch

from oumi.core.configs.params.fsdp_params import FSDPParams
from oumi.core.configs.params.training_params import TrainingParams
from oumi.core.distributed import cleanup_distributed, init_distributed, is_distributed
from oumi.core.tokenizers import BaseTokenizer
from oumi.core.trainers.oumi_trainer import Trainer
from oumi.datasets import DebugPretrainingDataset
from oumi.models import MLPEncoder


# Simple tokenizer (just a placeholder for this example)
class SimpleTokenizer(BaseTokenizer):
    def __init__(self):
        """Simple tokenizer for testing FSDP."""
        self.pad_token_id = 0

    def convert_ids_to_tokens(self, input_ids):
        """Converts input IDs to tokens."""
        return str(input_ids)

    def convert_tokens_to_ids(self, tokens):
        """Converts tokens to input IDs."""
        if isinstance(tokens, list):
            return [0] * len(tokens)
        return 0


# Test function
def test_fsdp_trainer():
    """Minimal FSDP loop."""
    # Initialize model and dataset

    if is_distributed():
        print("Initializing distributed process group")
        init_distributed()
    else:
        print("Not initializing distributed process group")

    #
    # Init model and dataset
    #
    model = MLPEncoder(input_dim=1024, hidden_dim=128, output_dim=1024)
    dataset = DebugPretrainingDataset(vocab_size=1024)
    tokenizer = SimpleTokenizer()

    #
    # Set up training parameters
    #
    fsdp_params = FSDPParams(enable_fsdp=True)

    training_params = TrainingParams(
        output_dir="output/fsdp_test_output",
        per_device_train_batch_size=32,
        num_train_epochs=2,
        max_steps=10,  # Just for quick testing
        save_steps=5,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_params,
        train_dataset=dataset,
        fsdp_params=fsdp_params,
    )

    #
    # Train session #1: from scratch
    #
    trainer.train()
    trainer.save_state()

    #
    # Train session #2: from checkpoint
    #
    new_model = MLPEncoder(input_dim=1024, hidden_dim=128, output_dim=1024)
    new_trainer = Trainer(
        model=new_model,
        processing_class=tokenizer,
        args=training_params,
        train_dataset=dataset,
        fsdp_params=fsdp_params,
    )
    # Resume training
    new_trainer.train(resume_from_checkpoint=training_params.output_dir)

    #
    # Test inference
    #
    test_input = torch.randint(low=0, high=1024, size=(10, 1024))
    test_input = test_input.to(new_trainer.device)
    with torch.no_grad():
        _output = new_trainer.model(test_input)

    if is_distributed():
        cleanup_distributed()

    return True


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Starting FSDP / DDP test with {world_size} GPUs")

    test_fsdp_trainer()
    print("FSDP test successful!")
