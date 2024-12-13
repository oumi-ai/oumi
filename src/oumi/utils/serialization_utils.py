import json

import torch


class TorchJsonEncoder(json.JSONEncoder):
    # Override default() method
    def default(self, obj):
        """Extending python's JSON Encoder to serialize torch dtype."""
        if obj is None:
            return ""
        elif isinstance(obj, torch.dtype):
            return str(obj)
        else:
            return super().default(obj)
