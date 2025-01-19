import numpy as np
import pytest
import torch

from oumi.builders import build_model
from oumi.core.configs import ModelParams
from oumi.core.registry import REGISTRY, RegistryType


def _convert_example_to_model_input(example: dict, device: torch.device) -> dict:
    return {
        key: (
            torch.from_numpy(value)
            if isinstance(value, np.ndarray)
            else torch.from_numpy(np.asarray(value))
        )
        .unsqueeze(0)
        .to(device, non_blocking=True)
        for key, value in example.items()
    }


@pytest.mark.parametrize(
    "from_registry",
    [False, True],
)
def test_instantiation_and_basic_usage(from_registry: bool):
    if from_registry:
        model_cls = REGISTRY.get("SimpleMnistCNN", RegistryType.MODEL)
        assert model_cls is not None
        model = model_cls()
    else:
        model_params = ModelParams(
            model_name="SimpleMnistCNN", load_pretrained_weights=False
        )
        model = build_model(model_params)

    model_device = next(model.parameters()).device

    for batch_size in (1, 2, 3):
        test_image = np.zeros(shape=(batch_size, 28, 28))

        outputs = model(
            **_convert_example_to_model_input(
                {"image": test_image}, device=model_device
            )
        )
        assert "logits" in outputs
