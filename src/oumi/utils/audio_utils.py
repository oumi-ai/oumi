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

"""Audio utility helpers for multimodal pipelines."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import requests

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import torch

try:  # pragma: no cover - optional dependency
    import torch as _torch
    import torchaudio as _torchaudio
    from torchaudio import functional as _ta_functional
except ImportError:  # pragma: no cover - handled at runtime
    _torch = None  # type: ignore[assignment]
    _torchaudio = None  # type: ignore[assignment]
    _ta_functional = None  # type: ignore[assignment]

_FILE_URL_PREFIX = "file://"


class AudioBackendUnavailableError(RuntimeError):
    """Raised when required audio dependencies are not available."""


def _ensure_torchaudio_available() -> None:
    if _torchaudio is None or _torch is None or _ta_functional is None:
        raise AudioBackendUnavailableError(
            "torchaudio and torch are required for audio preprocessing. "
            "Install them via `pip install torch torchaudio`."
        )


def _load_bytes_from_path(path: str | Path) -> bytes:
    if isinstance(path, str) and path.startswith(_FILE_URL_PREFIX):
        path = path[len(_FILE_URL_PREFIX) :]
    path = Path(path)
    if not path.is_file():
        raise ValueError(
            "Audio path is invalid: "
            f"{path} {'(not a file)' if path.exists() else '(does not exist)'}"
        )
    return path.read_bytes()


def _load_bytes_from_url(url: str, timeout: float = 30.0) -> bytes:
    if not url:
        raise ValueError("Empty audio URL")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failures
        raise RuntimeError(f"Failed to download audio from '{url}'") from exc
    return response.content


def load_audio_bytes(
    *,
    path: str | Path | None = None,
    url: str | None = None,
    binary: bytes | None = None,
) -> bytes:
    """Resolves raw audio bytes from one of the supported sources."""
    if binary is not None:
        if len(binary) == 0:
            raise ValueError("Audio binary payload is empty")
        return binary
    if path is not None:
        return _load_bytes_from_path(path)
    if url is not None:
        return _load_bytes_from_url(url)
    raise ValueError("No audio source provided")


def decode_audio_waveform(
    audio_bytes: bytes,
    *,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    """Decodes raw audio bytes into a waveform tensor."""
    _ensure_torchaudio_available()
    if audio_bytes is None or len(audio_bytes) == 0:
        raise ValueError("Empty audio bytes")

    buffer = io.BytesIO(audio_bytes)
    torchaudio_mod = cast(Any, _torchaudio)
    torch_mod = cast("torch", _torch)
    functional_mod = cast(Any, _ta_functional)

    waveform, sample_rate = torchaudio_mod.load(buffer)
    waveform = waveform.to(dtype=torch_mod.float32)

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be a positive integer")

    if sample_rate != target_sample_rate:
        waveform = functional_mod.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    return waveform, sample_rate


def encode_waveform_to_wav_bytes(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> bytes:
    """Serialises a waveform tensor to WAV bytes."""
    _ensure_torchaudio_available()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(
            f"Waveform tensor must be shaped (channels, samples). Got {waveform.shape}."
        )

    buffer = io.BytesIO()
    torchaudio_mod = cast(Any, _torchaudio)
    torchaudio_mod.save(buffer, waveform, sample_rate=sample_rate, format="wav")
    return buffer.getvalue()


def load_audio_wav_bytes(
    *,
    path: str | Path | None = None,
    url: str | None = None,
    binary: bytes | None = None,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> bytes:
    """Returns WAV bytes normalised to the target sample rate."""
    raw_bytes = load_audio_bytes(path=path, url=url, binary=binary)
    waveform, sample_rate = decode_audio_waveform(
        raw_bytes,
        target_sample_rate=target_sample_rate,
        mono=mono,
    )
    return encode_waveform_to_wav_bytes(waveform, sample_rate=sample_rate)


def waveform_to_numpy(waveform: torch.Tensor) -> np.ndarray:
    """Converts a waveform tensor to a NumPy array (channels x samples)."""
    _ensure_torchaudio_available()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    return waveform.detach().cpu().numpy()
