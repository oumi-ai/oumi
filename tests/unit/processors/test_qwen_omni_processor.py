import pytest

from oumi.core.processors.qwen_omni_processor import QwenOmniProcessorConfig


def test_qwen_omni_processor_config_defaults():
    config = QwenOmniProcessorConfig()
    assert config.audio_sample_rate == 16000
    assert config.audio_mono is True
    assert pytest.approx(config.video_fps, rel=0, abs=1e-6) == 2.0
