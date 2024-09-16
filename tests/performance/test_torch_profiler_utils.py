import tempfile

import numpy as np
import pytest
import torch
from torch import nn

from oumi.core.configs.params.profiler_params import ProfilerParams
from oumi.performance.torch_profiler_utils import torch_profile


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


@pytest.mark.parametrize(
    "params",
    [
        ProfilerParams(),
        ProfilerParams(
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            row_limit=-1,
        ),
        ProfilerParams(
            enable_cpu_profiling=True, enable_cuda_profiling=True, row_limit=2
        ),
        ProfilerParams(
            enable_cpu_profiling=True,
            enable_cuda_profiling=True,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            row_limit=2,
        ),
        ProfilerParams(
            enable_cpu_profiling=True,
            enable_cuda_profiling=False,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            row_limit=3,
        ),
    ],
)
def test_torch_profile(params: ProfilerParams):
    BATCH_SIZE = 256

    # Initialize a seed to improve test determinism.
    np.random.seed(42)
    # Generate random 3D points, centered around the origin (0,0,0)
    # using Gaussian distribution.
    x = np.random.normal(scale=0.8, size=(BATCH_SIZE, 3))
    targets = np.square(x)
    targets = targets[:, 0] + targets[:, 1] + targets[:, 2]
    # Define target as "within a unit sphere".
    targets = targets <= 1.0

    x = torch.from_numpy(x.astype(dtype=np.float32))
    targets = torch.from_numpy(
        np.reshape(targets, newshape=(BATCH_SIZE, 1)).astype(dtype=np.float32)
    )
    mlp = SimpleMLP()

    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adagrad(mlp.parameters(), lr=1e-2)

    with tempfile.TemporaryDirectory() as output_temp_dir:
        with torch_profile(params, training_output_dir=output_temp_dir):
            for epoch in range(4):
                optimizer.zero_grad()
                outputs = mlp(x)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
