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

"""Specialised processor for Qwen Omni multimodal models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as tvF
import transformers
from PIL import Image

from oumi.core.processors.default_processor import DefaultProcessor
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.core.types.conversation import ContentItem, Message, Type
from oumi.utils.audio_utils import decode_audio_waveform, waveform_to_numpy
from oumi.utils.conversation_utils import (
    base64encode_content_item_audio_bytes,
    base64encode_content_item_image_bytes,
    base64encode_content_item_video_bytes,
    load_audio_bytes_to_content_item,
    load_image_bytes_to_content_item,
    load_pil_image_from_content_item,
    load_video_bytes_to_content_item,
)
from oumi.utils.logging import logger
from oumi.utils.video_utils import decode_video_to_frames

try:  # pragma: no cover - optional dependency
    from qwen_omni_utils import process_mm_info as qwen_process_mm_info  # type: ignore

    _HAS_QWEN_OMNI_UTILS = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_QWEN_OMNI_UTILS = False


@dataclass
class QwenOmniProcessorConfig:
    """Configuration tweaks for Qwen Omni preprocessing."""

    audio_sample_rate: int = 16000
    audio_mono: bool = True
    video_fps: float = 2.0
    video_max_frames: int | None = None
    use_audio_in_video: bool = False
    preferred_video_backend: tuple[str, ...] = ("decord", "pyav")


class QwenOmniProcessor(DefaultProcessor):
    """Extends the default processor with audio/video handling for Qwen Omni."""

    def __init__(
        self,
        processor_name: str,
        worker_processor: Any,
        tokenizer: BaseTokenizer,
        *,
        label_ignore_index: int | None,
        ignore_features: list[str] | None = None,
        config: QwenOmniProcessorConfig | None = None,
    ) -> None:
        """Initialise the wrapped processor and multimodal configuration."""
        super().__init__(
            processor_name=processor_name,
            worker_processor=worker_processor,
            tokenizer=tokenizer,
            label_ignore_index=label_ignore_index,
            ignore_features=ignore_features,
        )
        self._config = config or QwenOmniProcessorConfig()

    @property
    def config(self) -> QwenOmniProcessorConfig:
        """Returns the processor configuration."""
        return self._config

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def prepare_multimodal_inputs(
        self,
        messages: Iterable[Message],
        *,
        use_audio_in_video: bool | None = None,
        return_video_kwargs: bool = False,
    ) -> tuple[Any, Any, Any, dict[str, Any] | None]:
        """Processes audio/image/video payloads for the Qwen processor."""
        qwen_messages = [
            self._convert_message_to_qwen_payload(message) for message in messages
        ]
        use_audio = (
            self._config.use_audio_in_video
            if use_audio_in_video is None
            else use_audio_in_video
        )

        if _HAS_QWEN_OMNI_UTILS:
            try:
                result = qwen_process_mm_info(
                    qwen_messages,
                    use_audio,
                    return_video_kwargs=return_video_kwargs,
                )
                if return_video_kwargs:
                    audios, images, videos, video_kwargs = result
                else:
                    audios, images, videos = result
                    video_kwargs = None
                return audios, images, videos, video_kwargs
            except Exception as exc:  # pragma: no cover - robustness
                logger.warning(
                    "qwen-omni-utils preprocessing failed; using built-in fallback: %s",
                    exc,
                )

        fallback = self._fallback_prepare_mm_inputs(
            messages,
            use_audio_in_video=use_audio,
            return_video_kwargs=return_video_kwargs,
        )
        return fallback

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image] | None = None,
        audios: list[Any] | None = None,
        videos: list[Any] | None = None,
        return_tensors: str | None = "pt",
        **kwargs: Any,
    ) -> Any:
        """Invoke the underlying processor with bundled multimodal inputs."""
        processor_kwargs: dict[str, Any] = dict(kwargs)
        inputs: dict[str, Any] = {}
        if images:
            inputs["images"] = images
        if audios:
            inputs["audio"] = audios
        if videos:
            inputs["videos"] = videos

        if inputs:
            inputs["text"] = text[0] if len(text) == 1 else text
        else:
            inputs["text"] = text

        result = self._worker_processor(
            return_tensors=return_tensors,
            **inputs,
            **processor_kwargs,
        )
        if result is None:
            raise RuntimeError("Processor returned `None`.")
        return self._normalise_processor_output(result, return_tensors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _normalise_processor_output(
        self, result: Any, return_tensors: str | None
    ) -> transformers.BatchEncoding:
        if result is None:
            raise RuntimeError("Processor returned `None`.")
        if isinstance(result, transformers.BatchFeature):
            for key in self.ignore_features:
                if key in result:
                    del result[key]
            return transformers.BatchEncoding(
                data=dict(**result), tensor_type=return_tensors
            )
        if isinstance(result, dict):
            return transformers.BatchEncoding(data=result, tensor_type=return_tensors)
        if isinstance(result, transformers.BatchEncoding):
            return result
        raise RuntimeError(
            "Processor returned an object that is not a BatchEncoding. "
            f"Actual type: {type(result)}"
        )

    def _convert_message_to_qwen_payload(self, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role.value}
        if isinstance(message.content, str):
            payload["content"] = message.content
            return payload

        qwen_items: list[dict[str, Any]] = []
        for item in message.content_items:
            if item.is_text():
                qwen_items.append({"type": "text", "text": item.content or ""})
            elif item.is_image():
                qwen_items.append(self._convert_image_item(item))
            elif item.is_audio():
                qwen_items.append(self._convert_audio_item(item))
            elif item.is_video():
                qwen_items.append(self._convert_video_item(item))
            else:  # pragma: no cover - defensive
                logger.warning("Unsupported content item type: %s", item.type)
        payload["content"] = qwen_items
        return payload

    def _convert_image_item(self, item: ContentItem) -> dict[str, Any]:
        if item.type == Type.IMAGE_PATH:
            return {"type": "image", "image": item.content}
        if item.type == Type.IMAGE_URL:
            return {"type": "image", "image_url": item.content}
        binary_item = load_image_bytes_to_content_item(item)
        data_url = base64encode_content_item_image_bytes(
            binary_item, add_mime_prefix=True
        )
        return {"type": "image", "image": data_url}

    def _convert_audio_item(self, item: ContentItem) -> dict[str, Any]:
        if item.type == Type.AUDIO_PATH:
            return {"type": "audio", "audio": item.content}
        if item.type == Type.AUDIO_URL:
            return {"type": "audio", "audio_url": item.content}
        binary_item = load_audio_bytes_to_content_item(
            item,
            target_sample_rate=self._config.audio_sample_rate,
            mono=self._config.audio_mono,
        )
        data_url = base64encode_content_item_audio_bytes(
            binary_item, add_mime_prefix=True
        )
        return {"type": "audio", "audio": data_url}

    def _convert_video_item(self, item: ContentItem) -> dict[str, Any]:
        if item.type == Type.VIDEO_PATH:
            return {"type": "video", "video": item.content}
        if item.type == Type.VIDEO_URL:
            return {"type": "video", "video_url": item.content}
        binary_item = load_video_bytes_to_content_item(item)
        data_url = base64encode_content_item_video_bytes(
            binary_item, add_mime_prefix=True
        )
        return {"type": "video", "video": data_url}

    def _fallback_prepare_mm_inputs(
        self,
        messages: Iterable[Message],
        *,
        use_audio_in_video: bool,
        return_video_kwargs: bool,
    ) -> tuple[Any, Any, Any, dict[str, Any] | None]:
        audios = self._fallback_audio(messages)
        images = self._fallback_images(messages)
        videos, video_fps = self._fallback_videos(messages)

        video_kwargs = {"fps": video_fps} if return_video_kwargs and video_fps else None
        if use_audio_in_video:
            logger.warning(
                "Audio-in-video extraction is unavailable in the fallback implementation;"
                " video audio will be ignored."
            )
        return audios, images, videos, video_kwargs

    def _fallback_audio(self, messages: Iterable[Message]) -> list[Any] | None:
        audio_arrays: list[Any] = []
        for message in messages:
            for item in message.audio_content_items:
                binary_item = load_audio_bytes_to_content_item(
                    item,
                    target_sample_rate=self._config.audio_sample_rate,
                    mono=self._config.audio_mono,
                )
                if not binary_item.binary:
                    raise ValueError("Audio content item is missing binary payload")
                waveform, _ = decode_audio_waveform(
                    binary_item.binary,
                    target_sample_rate=self._config.audio_sample_rate,
                    mono=self._config.audio_mono,
                )
                audio_arrays.append(waveform_to_numpy(waveform).squeeze(0))
        return audio_arrays or None

    def _fallback_images(self, messages: Iterable[Message]) -> list[Image.Image] | None:
        image_list: list[Image.Image] = []
        for message in messages:
            for item in message.image_content_items:
                image_item = item
                if image_item.type != Type.IMAGE_BINARY:
                    image_item = load_image_bytes_to_content_item(image_item)
                image_list.append(load_pil_image_from_content_item(image_item))
        return image_list or None

    def _fallback_videos(
        self,
        messages: Iterable[Message],
    ) -> tuple[list[torch.Tensor] | None, list[float] | None]:
        videos: list[torch.Tensor] = []
        fps_values: list[float] = []
        for message in messages:
            for item in message.video_content_items:
                binary_item = load_video_bytes_to_content_item(item)
                if not binary_item.binary:
                    raise ValueError("Video content item is missing binary payload")
                frames, fps = decode_video_to_frames(
                    binary_item.binary,
                    target_fps=self._config.video_fps,
                    max_frames=self._config.video_max_frames,
                    preferred_backend=self._config.preferred_video_backend,
                )
                if not frames:
                    continue
                if self._config.video_max_frames is not None:
                    frames = frames[: self._config.video_max_frames]
                tensor_frames = [tvF.pil_to_tensor(frame) for frame in frames]
                videos.append(torch.stack(tensor_frames).float())
                fps_values.append(fps)
        return (videos or None, fps_values or None)
