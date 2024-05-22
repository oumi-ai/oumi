from lema.evaluate import evaluate
from lema.train import train
from lema.utils.logging_utils import get_logger

logger = get_logger("lema")

__all__ = ["train", "evaluate"]
