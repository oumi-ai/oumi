import torch


def get_default_device_map_for_inference() -> str:
    return "cuda" if torch.cuda.is_available() else "auto"
