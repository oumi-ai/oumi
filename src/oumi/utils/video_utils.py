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

"""Video utility helpers for multimodal pipelines."""

from __future__ import annotations

import io
import tempfile
from collections.abc import Sequence
from pathlib import Path

import requests
from PIL import Image

from oumi.utils.logging import logger

try:  # pragma: no cover - optional dependency
    import decord  # type: ignore
except ImportError:  # pragma: no cover
    decord = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import av  # type: ignore
except ImportError:  # pragma: no cover
    av = None  # type: ignore

_FILE_URL_PREFIX = "file://"


class VideoBackendUnavailableError(RuntimeError):
    """Raised when no supported video decoding backend is available."""


def _load_bytes_from_path(path: str | Path) -> bytes:
    if isinstance(path, str) and path.startswith(_FILE_URL_PREFIX):
        path = path[len(_FILE_URL_PREFIX) :]
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            "Video path is invalid: "
            f"{path} {'(not a file)' if path.exists() else '(does not exist)'}"
        )
    return path.read_bytes()


def _load_bytes_from_url(url: str, timeout: float = 30.0) -> bytes:
    if not url:
        raise ValueError("Empty video URL")
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"Failed to download video from '{url}'") from exc
    return response.content


def load_video_bytes(
    *,
    path: str | Path | None = None,
    url: str | None = None,
    binary: bytes | None = None,
) -> bytes:
    """Resolves raw video bytes from one of the supported sources."""
    if binary is not None:
        if len(binary) == 0:
            raise ValueError("Video binary payload is empty")
        return binary
    if path is not None:
        return _load_bytes_from_path(path)
    if url is not None:
        return _load_bytes_from_url(url)
    raise ValueError("No video source provided")


def _resize_frame(frame: Image.Image, size: tuple[int, int] | None) -> Image.Image:
    if size is None:
        return frame
    width, height = size
    return frame.resize((width, height), Image.Resampling.BILINEAR)


def _decode_with_decord(
    video_bytes: bytes,
    *,
    target_fps: float | None,
    max_frames: int | None,
    frame_size: tuple[int, int] | None,
) -> tuple[list[Image.Image], float]:
    assert decord is not None  # noqa: S101 - validated by caller
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        vr = decord.VideoReader(tmp.name)
        num_frames = len(vr)
        orig_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else None

        if target_fps and orig_fps:
            step = max(1, int(round(orig_fps / target_fps)))
            effective_fps = orig_fps / step
        else:
            step = 1
            effective_fps = orig_fps or (target_fps or 0.0)

        indices = list(range(0, num_frames, step))
        if max_frames is not None:
            indices = indices[:max_frames]

        if not indices:
            return [], effective_fps

        frames_nd = vr.get_batch(indices).asnumpy()

    frames: list[Image.Image] = []
    for array in frames_nd:
        image = Image.fromarray(array)
        frames.append(_resize_frame(image, frame_size))
    return frames, effective_fps


def _decode_with_av(
    video_bytes: bytes,
    *,
    target_fps: float | None,
    max_frames: int | None,
    frame_size: tuple[int, int] | None,
) -> tuple[list[Image.Image], float]:
    if av is None:  # pragma: no cover - validated by caller
        raise VideoBackendUnavailableError(
            "PyAV is required for video decoding. Install via `pip install av`."
        )

    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    base_time: float | None = None
    target_interval = 1.0 / target_fps if target_fps and target_fps > 0 else None
    next_threshold = 0.0
    frames: list[Image.Image] = []
    effective_fps = (
        target_fps or float(stream.average_rate) if stream.average_rate else 0.0
    )

    for frame in container.decode(stream):
        timestamp = None
        if frame.time is not None:
            timestamp = float(frame.time)
        elif frame.pts is not None and frame.time_base is not None:
            timestamp = float(frame.pts * frame.time_base)

        if base_time is None:
            base_time = timestamp or 0.0

        relative_time = (timestamp or 0.0) - base_time
        if target_interval is not None:
            if relative_time + 1e-6 < next_threshold:
                continue
            next_threshold += target_interval
        frames.append(_resize_frame(frame.to_image(), frame_size))
        if max_frames is not None and len(frames) >= max_frames:
            break

    if target_interval is not None and len(frames) > 1:
        effective_fps = max(
            0.0, (len(frames) - 1) / (next_threshold or (1.0 / target_interval))
        )
    elif effective_fps is None:
        effective_fps = 0.0

    container.close()
    return frames, effective_fps


def decode_video_to_frames(
    video_bytes: bytes,
    *,
    target_fps: float | None = 2.0,
    max_frames: int | None = None,
    frame_size: tuple[int, int] | None = None,
    preferred_backend: Sequence[str] = ("decord", "pyav"),
) -> tuple[list[Image.Image], float]:
    """Decodes raw video bytes into a list of PIL frames.

    Returns the frames and the effective sampling FPS.
    """
    backends = list(preferred_backend)
    for backend in backends:
        if backend == "decord" and decord is not None:
            try:
                return _decode_with_decord(
                    video_bytes,
                    target_fps=target_fps,
                    max_frames=max_frames,
                    frame_size=frame_size,
                )
            except Exception as exc:  # pragma: no cover - backend failure
                logger.warning("Decord video decoding failed: %s", exc)
        if backend == "pyav" and av is not None:
            try:
                return _decode_with_av(
                    video_bytes,
                    target_fps=target_fps,
                    max_frames=max_frames,
                    frame_size=frame_size,
                )
            except Exception as exc:  # pragma: no cover - backend failure
                logger.warning("PyAV video decoding failed: %s", exc)
    raise VideoBackendUnavailableError(
        "No supported video decoding backend is available. Install `pyav` or `decord`."
    )
