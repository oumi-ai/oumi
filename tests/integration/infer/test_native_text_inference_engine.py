# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# pyright: reportGeneralTypeIssues=false

import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable

import jsonlines
import pytest
import transformers

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.processors.qwen_omni_processor import (
    QwenOmniProcessor,
    QwenOmniProcessorConfig,
)
from oumi.core.types.conversation import (
    ContentItem,
    Conversation,
    Message,
    Role,
    Type,
)
from oumi.inference import NativeTextInferenceEngine
from oumi.utils.image_utils import load_image_png_bytes_from_path
from tests.integration.infer import get_default_device_map_for_inference
from tests.markers import requires_cuda_initialized, requires_gpus


def _get_default_text_model_params() -> ModelParams:
    return ModelParams(
        model_name="openai-community/gpt2",
        trust_remote_code=True,
        chat_template="gpt2",
        tokenizer_pad_token="<|endoftext|>",
        device_map=get_default_device_map_for_inference(),
    )


def _get_default_image_model_params() -> ModelParams:
    return ModelParams(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        model_max_length=1024,
        trust_remote_code=True,
        chat_template="qwen2-vl-instruct",
        device_map=get_default_device_map_for_inference(),
    )


def _get_default_inference_config() -> InferenceConfig:
    return InferenceConfig(
        generation=GenerationParams(
            max_new_tokens=5, use_sampling=False, temperature=0.0, min_p=0.0, seed=42
        )
    )


def _setup_input_conversations(filepath: str, conversations: list[Conversation]):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).touch()
    with jsonlines.open(filepath, mode="w") as writer:
        for conversation in conversations:
            json_obj = conversation.to_dict()
            writer.write(json_obj)
    # Add some empty lines into the file
    with open(filepath, "a") as f:
        f.write("\n\n\n")


#
# Tests
#
@requires_gpus()
def test_infer_online():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
    conversation = Conversation(
        messages=[
            Message(
                content="Hello world!",
                role=Role.USER,
            ),
            Message(
                content="Hello again!",
                role=Role.USER,
            ),
        ],
        metadata={"foo": "bar"},
        conversation_id="123",
    )
    expected_result = [
        Conversation(
            messages=[
                *conversation.messages,
                Message(
                    content="The first time I saw",
                    role=Role.ASSISTANT,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
    ]
    result = engine.infer([conversation], _get_default_inference_config())
    assert expected_result == result


def test_infer_online_empty():
    engine = NativeTextInferenceEngine(_get_default_text_model_params())
    result = engine.infer([], _get_default_inference_config())
    assert [] == result


def test_infer_online_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)
        result = engine.infer(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


#
# Multimodal aggregation tests for Qwen Omni processors
#


class _MockOmniProcessor(QwenOmniProcessor):
    def __init__(
        self,
        prepare_return: tuple[Any, Any, Any, dict[str, Any] | None],
        *,
        config: QwenOmniProcessorConfig | None = None,
        prepare_hook: Callable[..., None] | None = None,
    ) -> None:
        # Deliberately bypass parent initialiser to avoid heavy dependencies.
        self.prepare_return = prepare_return
        self.prepare_hook = prepare_hook
        self._config = config or QwenOmniProcessorConfig()
        self.last_kwargs: dict[str, Any] | None = None
        self.prepare_args: dict[str, Any] | None = None

    def prepare_multimodal_inputs(
        self,
        messages: Iterable[Message],
        *,
        use_audio_in_video: bool | None,
        return_video_kwargs: bool,
    ) -> tuple[Any, Any, Any, dict[str, Any] | None]:
        if self.prepare_hook:
            self.prepare_hook(messages, use_audio_in_video, return_video_kwargs)
        self.prepare_args = {
            "messages": list(messages),
            "use_audio_in_video": use_audio_in_video,
            "return_video_kwargs": return_video_kwargs,
        }
        return self.prepare_return

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Any] | None = None,
        audios: list[Any] | None = None,
        videos: list[Any] | None = None,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> transformers.BatchEncoding:
        self.last_kwargs = {
            "text": text,
            "images": images,
            "audios": audios,
            "videos": videos,
            "return_tensors": return_tensors,
            **kwargs,
        }
        return transformers.BatchEncoding(data={"dummy": []})

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        return ""


def _make_mock_omni_engine(processor: _MockOmniProcessor) -> NativeTextInferenceEngine:
    engine = NativeTextInferenceEngine.__new__(NativeTextInferenceEngine)
    engine._processor = processor  # type: ignore[attr-defined]
    engine._supports_multiple_images = True  # type: ignore[attr-defined]
    return engine


def _make_multi_image_conversation() -> Conversation:
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_PATH, content="image_a.png"),
                    ContentItem(type=Type.IMAGE_PATH, content="image_b.png"),
                ],
            )
        ]
    )


def test_qwen_omni_multi_image_aggregation():
    mock_processor = _MockOmniProcessor(
        prepare_return=(None, ["img_a", "img_b"], None, None)
    )
    engine = _make_mock_omni_engine(mock_processor)
    text_prompts = ["prompt"]
    conversation = _make_multi_image_conversation()

    engine._generate_batch_encoding_with_processor(text_prompts, [conversation])

    assert mock_processor.prepare_args is not None
    assert (
        mock_processor.prepare_args["use_audio_in_video"]
        == mock_processor.config.use_audio_in_video
    )
    assert mock_processor.last_kwargs is not None
    assert mock_processor.last_kwargs["images"] == ["img_a", "img_b"]
    assert mock_processor.last_kwargs["audios"] is None
    assert mock_processor.last_kwargs["videos"] is None


def test_qwen_omni_interleaved_images_and_text():
    observed_sequences: list[list[Type]] = []

    def _hook(messages: Iterable[Message], *_: Any) -> None:
        for message in messages:
            observed_sequences.append([item.type for item in message.content_items])

    mock_processor = _MockOmniProcessor(
        prepare_return=(None, ["img_a", "img_b"], None, None),
        prepare_hook=_hook,
    )
    engine = _make_mock_omni_engine(mock_processor)
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.IMAGE_PATH, content="image_a.png"),
                    ContentItem(type=Type.TEXT, content="Describe this"),
                    ContentItem(type=Type.IMAGE_PATH, content="image_b.png"),
                ],
            )
        ]
    )

    engine._generate_batch_encoding_with_processor(["prompt"], [conversation])

    assert observed_sequences == [[Type.IMAGE_PATH, Type.TEXT, Type.IMAGE_PATH]], (
        "Message order should be preserved when preparing multimodal inputs."
    )


def test_qwen_omni_video_with_audio():
    config = QwenOmniProcessorConfig(use_audio_in_video=True)
    mock_processor = _MockOmniProcessor(
        prepare_return=(
            ["audio_array"],
            None,
            ["video_frames"],
            {"fps": [2.0]},
        ),
        config=config,
    )
    engine = _make_mock_omni_engine(mock_processor)
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.VIDEO_PATH, content="video.mp4"),
                    ContentItem(type=Type.AUDIO_PATH, content="audio.wav"),
                ],
            )
        ]
    )

    engine._generate_batch_encoding_with_processor(["prompt"], [conversation])

    assert mock_processor.last_kwargs is not None
    assert mock_processor.last_kwargs["audios"] == ["audio_array"]
    assert mock_processor.last_kwargs["videos"] == ["video_frames"]
    assert mock_processor.last_kwargs["mm_processor_kwargs"] == {"fps": [2.0]}
    assert mock_processor.last_kwargs["use_audio_in_video"] is True


def test_qwen_omni_audio_only():
    mock_processor = _MockOmniProcessor(
        prepare_return=(["audio_array"], None, None, None)
    )
    engine = _make_mock_omni_engine(mock_processor)
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    ContentItem(type=Type.AUDIO_PATH, content="audio.wav"),
                ],
            )
        ]
    )

    engine._generate_batch_encoding_with_processor(["prompt"], [conversation])

    assert mock_processor.last_kwargs is not None
    assert mock_processor.last_kwargs["audios"] == ["audio_array"]
    assert mock_processor.last_kwargs["images"] is None
    assert mock_processor.last_kwargs["videos"] is None


def test_infer_from_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        conversation = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation])
        expected_result = [
            Conversation(
                messages=[
                    *conversation.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            )
        ]
        config = _get_default_inference_config()
        config.input_path = str(input_path)
        result = engine.infer(inference_config=config)
        assert expected_result == result


def test_infer_from_file_empty():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [])
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        inference_config = _get_default_inference_config()
        inference_config.input_path = str(input_path)
        result = engine.infer(inference_config=inference_config)
        assert [] == result


def test_infer_from_file_to_file():
    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_text_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    content="Hello world!",
                    role=Role.USER,
                ),
                Message(
                    content="Hello again!",
                    role=Role.USER,
                ),
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    content="Touche!",
                    role=Role.USER,
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1, conversation_2])
        expected_result = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="The first time I saw",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="The U.S.",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)
        result = engine.infer(
            [conversation_1, conversation_2],
            inference_config,
        )
        assert result == expected_result
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert expected_result == parsed_conversations


@requires_cuda_initialized()
@pytest.mark.single_gpu
def test_infer_from_file_to_file_with_images(root_testdata_dir: Path):
    png_image_bytes_great_wave = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "the_great_wave_off_kanagawa.jpg"
    )
    png_image_bytes_logo = load_image_png_bytes_from_path(
        root_testdata_dir / "images" / "oumi_logo_dark.png"
    )

    test_prompt: str = "Generate a short, descriptive caption for this image!"

    with tempfile.TemporaryDirectory() as output_temp_dir:
        engine = NativeTextInferenceEngine(_get_default_image_model_params())
        conversation_1 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=png_image_bytes_great_wave,
                        ),
                        ContentItem(
                            type=Type.TEXT,
                            content=test_prompt,
                        ),
                    ],
                )
            ],
            metadata={"foo": "bar"},
            conversation_id="123",
        )
        conversation_2 = Conversation(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        ContentItem(
                            type=Type.IMAGE_BINARY,
                            binary=png_image_bytes_logo,
                        ),
                        ContentItem(
                            type=Type.TEXT,
                            content=test_prompt,
                        ),
                    ],
                ),
            ],
            metadata={"umi": "bar"},
            conversation_id="133",
        )
        input_path = Path(output_temp_dir) / "foo" / "input.jsonl"
        _setup_input_conversations(str(input_path), [conversation_1, conversation_2])

        output_path = Path(output_temp_dir) / "b" / "output.jsonl"
        inference_config = _get_default_inference_config()
        inference_config.output_path = str(output_path)

        result = engine.infer(
            [conversation_1, conversation_2],
            inference_config,
        )
        with open(output_path) as f:
            parsed_conversations = []
            for line in f:
                parsed_conversations.append(Conversation.from_json(line))
            assert result == parsed_conversations

        expected_results = [
            Conversation(
                messages=[
                    *conversation_1.messages,
                    Message(
                        content="",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"foo": "bar"},
                conversation_id="123",
            ),
            Conversation(
                messages=[
                    *conversation_2.messages,
                    Message(
                        content="",
                        role=Role.ASSISTANT,
                    ),
                ],
                metadata={"umi": "bar"},
                conversation_id="133",
            ),
        ]
        # Verify that the model response isn't empty, and verify that the results
        # are as expected except for the response content.
        assert len(result) == len(expected_results)
        for expected, actual in zip(expected_results, result):
            assert actual.messages[-1].content
            actual_dict = actual.to_dict()
            actual_dict["messages"][-1]["content"] = ""
            actual = Conversation.from_dict(actual_dict)
            assert actual == expected


def test_unsupported_model_raises_error():
    model_params = ModelParams(
        model_name="MlpEncoder",
        tokenizer_name="gpt2",
        tokenizer_pad_token="<|endoftext|>",
        load_pretrained_weights=False,
    )
    with pytest.raises(ValueError, match="requires a generation config"):
        NativeTextInferenceEngine(model_params)
