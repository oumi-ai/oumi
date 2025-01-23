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
        ).to(device, non_blocking=True)
        for key, value in example.items()
    }


@pytest.mark.parametrize(
    "from_registry",
    [False, True],
)
def test_instantiation_and_basic_usage(from_registry: bool):
    if from_registry:
        model_cls = REGISTRY.get("CnnClassifier", RegistryType.MODEL)
        assert model_cls is not None
        model = model_cls(image_width=28, image_height=28, in_channels=1, output_dim=10)
    else:
        model_params = ModelParams(
            model_name="CnnClassifier",
            load_pretrained_weights=False,
            model_kwargs={
                "image_width": 28,
                "image_height": 28,
                "in_channels": 1,
                "output_dim": 10,
            },
        )
        model = build_model(model_params)

    model_device = next(model.parameters()).device

    for with_label in (False, True):
        for batch_size in (1, 2, 3):
            test_tag = f"bs={batch_size}, with_label: {with_label}"

            test_image = np.zeros(shape=(batch_size, 1, 28, 28), dtype=np.float32)

            inputs: dict = {"images": test_image}
            if with_label:
                inputs["labels"] = [4] * batch_size

            with torch.no_grad():
                outputs = model(
                    **_convert_example_to_model_input(inputs, device=model_device)
                )
            assert outputs.keys() == ({"logits", "loss"} if with_label else {"logits"})
            assert isinstance(outputs["logits"], torch.Tensor), test_tag
            logits = outputs["logits"].cpu().numpy()
            assert logits.shape == (batch_size, 10), test_tag
            assert logits.dtype == np.float32, test_tag
            if with_label:
                loss = outputs["loss"].cpu().numpy()
                assert loss.shape == (), test_tag
                assert loss.dtype == np.float32, test_tag
