import torch


def get_default_device_map_for_inference() -> str:
    return (
        "cuda"
        if (torch.cuda.is_available() and torch.cuda.device_count() > 1)
        else "auto"
    )
