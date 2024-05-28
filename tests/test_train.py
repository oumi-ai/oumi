import os
import pytest
import tempfile

from lema.train import TrainingConfig
from omegaconf import OmegaConf

from lema import train


def test_basic_train():
    config: TrainingConfig = TrainingConfig()
    # train(config)