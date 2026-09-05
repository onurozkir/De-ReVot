"""Audio resampler and format converter (48kHz <-> 16kHz <-> 24kHz)."""

from __future__ import annotations

import numpy as np

from voice_translator.audio.signal import pcm_to_float32

try:
    import soxr  # High quality & fast
except ImportError:
    soxr = None  # type: ignore

try:
    from scipy import signal
except ImportError:
    signal = None  # type: ignore


class AudioResampler:
    """Stateful resampler and channel converter."""

    def __init__(
        self,
        in_rate: int,
        out_rate: int,
        in_channels: int = 1,
        out_channels: int = 1,
        streaming: bool = False,
    ):
        self.in_rate = int(in_rate)
        self.out_rate = int(out_rate)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.ratio = self.out_rate / self.in_rate if self.in_rate > 0 else 1.0
        self.streaming = streaming
        self._soxr_stream = None

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio channels, dtype to float32, and resample to target rate."""
        if len(audio) == 0:
            return np.empty(0, dtype=np.float32)

        # 1. Convert to float32 in [-1.0, 1.0] if integer
        audio = pcm_to_float32(audio)

        # 2. Channel conversion
        if audio.ndim == 2:
            if self.out_channels == 1:
                audio = np.mean(audio, axis=1)
        elif audio.ndim == 1 and self.out_channels > 1:
            audio = np.tile(audio[:, np.newaxis], (1, self.out_channels))

        # 3. Resampling
        if self.in_rate == self.out_rate:
            return audio

        if soxr is not None:
            if self.streaming and hasattr(soxr, "ResampleStream"):
                if self._soxr_stream is None:
                    self._soxr_stream = soxr.ResampleStream(
                        self.in_rate,
                        self.out_rate,
                        self.out_channels,
                        dtype="float32",
                        quality="HQ",
                    )
                return np.asarray(self._soxr_stream.resample_chunk(audio, last=False), dtype=np.float32)
            return np.asarray(soxr.resample(audio, self.in_rate, self.out_rate), dtype=np.float32)

        if signal is not None:
            num_out_samples = int(round(len(audio) * self.ratio))
            return signal.resample(audio, num_out_samples).astype(np.float32)

        # Fallback linear interpolation
        old_indices = np.linspace(0, len(audio) - 1, len(audio))
        new_len = int(round(len(audio) * self.ratio))
        new_indices = np.linspace(0, len(audio) - 1, new_len)
        return np.interp(new_indices, old_indices, audio).astype(np.float32)

    def flush(self) -> np.ndarray:
        """Emit delayed samples from a streaming soxr instance and reset its state."""
        if self._soxr_stream is None:
            return np.empty(0, dtype=np.float32)
        empty = (
            np.empty(0, dtype=np.float32)
            if self.out_channels == 1
            else np.empty((0, self.out_channels), dtype=np.float32)
        )
        tail = np.asarray(self._soxr_stream.resample_chunk(empty, last=True), dtype=np.float32)
        self._soxr_stream.clear()
        self._soxr_stream = None
        return tail
