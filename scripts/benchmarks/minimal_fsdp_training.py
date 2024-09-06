import torch
import torch.nn as nn
from torch.utils.data import Dataset

from lema.core.configs.params.fsdp_params import FSDPParams
from lema.core.configs.params.training_params import TrainingParams
from lema.core.tokenizers import BaseTokenizer
from lema.core.trainers.lema_trainer import Trainer


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        """Simple model for testing FSDP."""
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        """Forward pass."""
        return self.linear(x)


# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        """Simple dataset for testing FSDP."""
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Gets the sample at index `idx`."""
        return {"input_ids": self.data[idx], "labels": self.labels[idx]}


# Simple tokenizer (just a placeholder for this example)
class SimpleTokenizer(BaseTokenizer):
    def __init__(self):
        """Simple tokenizer for testing FSDP."""
        self.pad_token_id = 0


# Test function
def test_fsdp_trainer():
    """Minimal FSDP loop."""
    # Initialize model and dataset
    model = SimpleModel()
    dataset = SimpleDataset()

    # Set up FSDP parameters
    fsdp_params = FSDPParams(enable_fsdp=True)

    # Set up training parameters
    training_params = TrainingParams(
        output_dir="./fsdp_test_output",
        per_device_train_batch_size=32,
        num_train_epochs=2,
        max_steps=10,  # Just for quick testing
        save_steps=5,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        tokenizer=SimpleTokenizer(),
        args=training_params,
        train_dataset=dataset,
        fsdp_params=fsdp_params,
    )

    # Train for a few steps
    trainer.train()

    # Save the model
    trainer.save_state()

    # Load the model from checkpoint
    new_trainer = Trainer(
        model=SimpleModel(),
        tokenizer=SimpleTokenizer(),
        args=training_params,
        train_dataset=dataset,
        fsdp_params=fsdp_params,
    )
    new_trainer._load_from_checkpoint("./fsdp_test_output")

    # Resume training
    new_trainer.train()

    # Test inference
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = new_trainer.model(test_input)

    print("Test output:", output)

    return True


# Run the test
if __name__ == "__main__":
    success = test_fsdp_trainer()
    print("FSDP test successful:", success)
